///////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cumem.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

#if !defined(WIN32) && defined(_MSC_VER)
#define WIN32
#endif

#ifdef WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

#define CUMEM_DEFAULT_GRANULARITY 512
//#define CUMEM_DEBUG

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cumem {

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CUMEM_CHECK(call) do { \
  cumemStatus_t status = call; \
  if( status != CUMEM_STATUS_SUCCESS ) { \
    return status; \
  } \
} while(0)

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CUMEM_CHECK_OR_UNLOCK_AND_RETURN(call, lock) do { \
  cumemStatus_t status = call; \
  if( status != CUMEM_STATUS_SUCCESS ) { \
    lock.unlock(); \
    return status; \
  } \
} while(0)

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK(cond, error) do { \
  if( !(cond) ) { \
    return error; \
  } \
} while(0)

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
  cudaError_t cuda_error = call; \
  if( cuda_error == cudaErrorMemoryAllocation ) { \
    return CUMEM_STATUS_OUT_OF_MEMORY; \
  } \
  else if( cuda_error != cudaSuccess ) { \
    return CUMEM_STATUS_CUDA_ERROR; \
  } \
} while(0)

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#define CHECK_WIN32(call, error_code) do { \
  SetLastError(0); /* Clean the flag. */ \
  call; \
  DWORD status = GetLastError(); \
  if( status ) \
    return error_code; \
} while(0)
#else
#define CHECK_PTHREAD(call, error_code) do { \
  int status = call; \
  if( status ) { \
    return error_code; \
  } \
} while(0)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

class Mutex
{
#ifdef WIN32
  CRITICAL_SECTION m_critical_section;
#else
  pthread_mutex_t  m_mutex;
#endif

public:
  /// Initialize the mutex.
  cumemStatus_t initialize();
  /// Finalize the mutex.
  cumemStatus_t finalize();
  /// Lock the mutex.
  cumemStatus_t lock();
  /// Unlock the mutex.
  cumemStatus_t unlock();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Mutex::initialize()
{
#ifdef WIN32
  CHECK_WIN32(InitializeCriticalSection(&m_critical_section), CUMEM_STATUS_UNKNOWN_ERROR);
#else
  // pthread_mutexattr_t attr;
  // CHECK_PTHREAD_OR_THROW(pthread_mutexattr_init(&attr), CUMEM_STATUS_UNKNOWN_ERROR);
  // CHECK_PTHREAD_OR_THROW(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE), CUMEM_STATUS_UNKNOWN_ERROR);
  // CHECK_PTHREAD_OR_THROW(pthread_mutex_init(&m_mutex, &attr), CUMEM_STATUS_UNKNOWN_ERROR);
  CHECK_PTHREAD(pthread_mutex_init(&m_mutex, NULL), CUMEM_STATUS_UNKNOWN_ERROR);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Mutex::finalize()
{
#ifdef WIN32
  CHECK_WIN32(DeleteCriticalSection(&m_critical_section), CUMEM_STATUS_UNKNOWN_ERROR);
#else
  CHECK_PTHREAD(pthread_mutex_destroy(&m_mutex), CUMEM_STATUS_UNKNOWN_ERROR);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Mutex::lock()
{
#ifdef WIN32
  CHECK_WIN32(EnterCriticalSection(&m_critical_section), CUMEM_STATUS_UNKNOWN_ERROR);
#else
  CHECK_PTHREAD(pthread_mutex_lock(&m_mutex), CUMEM_STATUS_UNKNOWN_ERROR);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Mutex::unlock()
{
#ifdef WIN32
  CHECK_WIN32(LeaveCriticalSection(&m_critical_section), CUMEM_STATUS_UNKNOWN_ERROR);
#else
  CHECK_PTHREAD(pthread_mutex_unlock(&m_mutex), CUMEM_STATUS_UNKNOWN_ERROR);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class Lock
{
  /// The mutex.
  Mutex *m_mutex;
  
public:
  /// Ctor.
  Lock() : m_mutex(NULL) {}
  /// Lock the mutex.
  inline cumemStatus_t lock(Mutex *mutex) { m_mutex = mutex; return m_mutex->lock(); }
  /// Unlock the mutex.
  inline cumemStatus_t unlock() { return m_mutex->unlock(); }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

class Block
{
  /// The pointer to the memory region on the device. 
  char *m_data;
  /// The size of the memory buffer.
  std::size_t m_size;
  /// The prev/next blocks in the linked list of blocks.
  Block *m_next;
  /// Is it a head node (i.e. a node obtained from parent->allocate or cudaMalloc).
  bool m_is_head;

public:
  /// Create a block.
  Block(char *data, std::size_t size, Block *next, bool is_head)
    : m_data(data)
    , m_size(size)
    , m_next(next)
    , m_is_head(is_head)
  {}
  
  /// The data.
  inline const char* get_data() const { return m_data; }
  /// The data (mutable).
  inline char* get_data() { return m_data; }
  
  /// The size of the block.
  inline std::size_t get_size() const { return m_size; }

  /// The next block in the linked list.
  inline const Block* get_next() const { return m_next; }
  /// The next block in the linked list (mutable).
  inline Block* get_next() { return m_next; }
  
  /// Is it a head block.
  inline bool is_head() const { return m_is_head; }

  /// Change the next block.
  inline void set_next(Block *next) { m_next = next; }
  /// Change the size of the block.
  inline void set_size(std::size_t size) { m_size = size; }
  /// Set the head flag.
  inline void set_head_flag(bool is_head) { m_is_head = is_head; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

class Manager
{
  /// The parent manager.
  Manager *m_parent;
  /// The children managers.
  std::vector<Manager> m_children;
  /// The GPU device where the memory is allocated.
  int m_device;
  /// The stream this manager is associated with. It could be NULL.
  cudaStream_t m_stream;
  /// The list of used blocks.
  Block *m_used_blocks;
  /// The list of free blocks.
  Block *m_free_blocks;
  /// The managed memory size.
  std::size_t m_size;
  /// The memory allocation granularity
  std::size_t m_granularity;
  /// The flags.
  unsigned m_flags;
  
  /// To support multi-threading. Each manager has its own mutex.
  Mutex m_mutex;

public:
  /// Create an unitialized manager.
  Manager();
  /// Dtor.
  ~Manager();

  /// Allocate a block of memory.
  cumemStatus_t allocate(void *&ptr, std::size_t size);
  /// Release a block of memory.
  cumemStatus_t release(void *ptr);
  /// Release memory. It returns true if we have no memory leak.
  cumemStatus_t release_all(bool &memory_leak);
  /// Reserve memory for a manager.
  cumemStatus_t reserve(std::size_t size);
  /// Steal memory from another manager.
  cumemStatus_t steal(void *&ptr, std::size_t size);

  /// Print the list of free blocks.
  inline std::size_t print_free_blocks(FILE *file) const 
  { 
    return print_list(file, "free", m_free_blocks); 
  }
  /// Print the list of used blocks.
  inline std::size_t print_used_blocks(FILE *file) const 
  { 
    return print_list(file, "used", m_used_blocks); 
  }

  /// The root manager for a given device.
  static inline Manager& get_root_manager(int device) { return get_root_managers()[device]; }
  /// The list of all the root managers.
  static std::vector<Manager>& get_root_managers();

  /// The amount of used memory.
  inline std::size_t get_used_memory() const { return get_memory(m_used_blocks); }
  /// The amount of used memory.
  inline std::size_t get_free_memory() const { return get_memory(m_free_blocks); }
  
  /// The children.
  inline std::vector<Manager>& get_children() { return m_children; }
  /// The children.
  inline const std::vector<Manager>& get_children() const { return m_children; }
  /// Get a specific child based on the stream id.
  cumemStatus_t get_child(Manager *&manager, cudaStream_t stream);
  /// The associated device.
  inline int get_device() const { return m_device; }
  /// The flags.
  inline unsigned get_flags() const { return m_flags; }
  /// Get the mutex.
  inline Mutex* get_mutex() { return &m_mutex; }
  /// The size allocated to that manager.
  inline std::size_t get_size() const { return m_size; }
  /// The CUDA stream.
  inline cudaStream_t get_stream() const { return m_stream; }
  /// The allocation granularity.
  inline size_t get_granularity() const { return m_granularity; }
  
  /// Define the parent.
  inline void set_parent(Manager *parent) { m_parent = parent; }
  /// Define the device.
  inline void set_device(int device) { m_device = device; }
  /// Define the stream.
  inline void set_stream(cudaStream_t stream) { m_stream = stream; }
  /// Define the flags.
  inline void set_flags(unsigned flags) { m_flags = flags; }
  /// Define the granularity
  inline void set_granularity(unsigned granularity) { m_granularity = granularity; }
  
private:
  /// Allocate a new block and add it to the free list.
  cumemStatus_t allocate_block(Block *&curr, Block *&prev, std::size_t size);
  /// Release a block from the active list.
  cumemStatus_t release_block(Block *curr, Block *prev);
  /// Find the best free node based on the size.
  cumemStatus_t find_best_block(Block *&curr, Block *&prev, std::size_t size);
  /// Extract a node from the list of free blocks.
  cumemStatus_t extract_block(Block *curr, Block *prev, std::size_t size, bool stolen);
  
  /// Give a free block from that manager.
  cumemStatus_t give_block(void *&data, std::size_t &data_size, std::size_t size);
  /// Steal a block from another manager.
  cumemStatus_t steal_block(void *&data, std::size_t &data_size, std::size_t size);
  
  /// The memory consumption of a list.
  std::size_t get_memory(const Block *head) const;
  /// Print an internal linked list.
  std::size_t print_list(FILE *file, const char *name, const Block *head) const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

Manager::Manager()
  : m_parent(NULL)
  , m_children()
  , m_device(-1)
  , m_stream(NULL)
  , m_used_blocks(NULL)
  , m_free_blocks(NULL)
  , m_size(0)
  , m_granularity(CUMEM_DEFAULT_GRANULARITY)
  , m_flags(CUMEM_FLAGS_DEFAULT)
  , m_mutex()
{}

///////////////////////////////////////////////////////////////////////////////////////////////////

Manager::~Manager()
{
  bool memory_leak;
  release_all(memory_leak);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::allocate(void *&ptr, std::size_t size)
{
  // Lock to make sure only one thread execute that fragment of code.
  Lock lock;
  CUMEM_CHECK(lock.lock(&m_mutex));

  // Find the best fit.
  Block *best = NULL, *prev = NULL;
  CUMEM_CHECK_OR_UNLOCK_AND_RETURN(find_best_block(best, prev, size), lock);

  // If there's no block left in the list of free blocks (with a sufficient size). Request a new block. 
  if( best == NULL && !(m_flags & CUMEM_FLAGS_CANNOT_GROW) )
  {
    CUMEM_CHECK_OR_UNLOCK_AND_RETURN(allocate_block(best, prev, size), lock);
  }
  
  // Make sure we do have a block or quit.
  if( !best )
  {
    CUMEM_CHECK(lock.unlock());
    ptr = NULL;
    return CUMEM_STATUS_OUT_OF_MEMORY;
  }

  // Split the free block if needed.
  CUMEM_CHECK_OR_UNLOCK_AND_RETURN(extract_block(best, prev, size, false), lock);

  // Push the node to the list of used nodes.
  best->set_next(m_used_blocks);
  m_used_blocks = best;

  // Return the new pointer into memory.
  CUMEM_CHECK(lock.unlock());
  ptr = m_used_blocks->get_data();
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::allocate_block(Block *&curr, Block *&prev, std::size_t size)
{
  // Reset the outputs.
  curr = prev = NULL;

  // Try to allocate data from the parent or the device.
  void *data = NULL;
  if( m_parent )
    CUMEM_CHECK(m_parent->allocate(data, size));
  else
  {
    if( m_flags & CUMEM_FLAGS_USE_UNIFIED_MEM )
    {
      #ifdef CUMEM_DEBUG
      std::cout << "attempting cudaMallocManaged of size " << size << "B" << std::endl;
      #endif
      if( m_flags & CUMEM_FLAGS_MEM_ATTACH_HOST )
      {
        CHECK_CUDA(cudaMallocManaged(&data, size, cudaMemAttachHost));
      }
      else
      {
        CHECK_CUDA(cudaMallocManaged(&data, size, cudaMemAttachGlobal));
      }
      #ifdef CUMEM_DEBUG
      std::cout << "cudaMallocManaged of size " << size << "B address=" << (void*)data << std::endl;
      #endif
    }
    else
    {
      CHECK_CUDA(cudaSetDevice(m_device));
      #ifdef CUMEM_DEBUG
      std::cout << "attempting cudaMalloc of size " << size << "B" << std::endl;
      #endif
      CHECK_CUDA(cudaMalloc(&data, size));
      #ifdef CUMEM_DEBUG
      std::cout << "cudaMalloc of size " << size << "B address=" << (void*)data << std::endl;
      #endif
    }
  }
  
  // If it failed, there's an unexpected issue.
  assert(data);

  // We have data, we now need to add it to the list of free nodes. We keep the list sorted.
  Block *next = m_free_blocks;
  for( ; next && next->get_data() < data ; next = next->get_next() )
    prev = next;
  curr = new Block((char*) data, size, next, true);
  if( prev )
    prev->set_next(curr);
  else
    m_free_blocks = curr;
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::extract_block(Block *curr, Block *prev, std::size_t size, bool stolen)
{
  // We have two cases: 1/ It is the right size so we keep it or 2/ it is too large and we split the node.
  Block *next;
  if( curr->get_size() == size )
    next = curr->get_next();
  else
  {
    std::size_t remaining = curr->get_size()-size;
    Block *new_block = new Block(curr->get_data() + size, remaining, curr->get_next(), stolen);
    next = new_block;
    curr->set_size(size);
  }
  
  // Redo the "branching" in the nodes.
  if( prev )
    prev->set_next(next);
  else
    m_free_blocks = next;
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::find_best_block(Block *&best, Block *&prev, std::size_t size)
{
  best = NULL, prev = NULL;
  for( Block *temp = m_free_blocks, *temp_prev = NULL ; temp ; temp = temp->get_next() )
  {
    if( temp->get_size() >= size && (!best || temp->get_size() < best->get_size()) )
    {
      best = temp;
      prev = temp_prev;
    }
    temp_prev = temp;
  }
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::get_child(Manager *&manager, cudaStream_t stream)
{
  for( std::size_t i = 0 ; i < m_children.size() ; ++i )
    if( m_children[i].m_stream == stream )
    {
      manager = &m_children[i];
      return CUMEM_STATUS_SUCCESS;
    }
  return CUMEM_STATUS_INVALID_ARGUMENT;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t Manager::get_memory(const Block *head) const
{
  std::size_t size = 0;
  for( Block *curr = (Block*) head ; curr ; curr = curr->get_next() )
    size += curr->get_size();
  return size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Manager>& Manager::get_root_managers()
{
  static std::vector<Manager> managers;
  return managers;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::give_block(void *&block_data, std::size_t &block_size, std::size_t size)
{
  // Make sure the block is not in use any more. It could be too coarse grain and we may change 
  // it in the future.
  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  
  // Init the returned values to 0.
  block_data = NULL;
  block_size = 0;
  
  // Find the best node to steal and reserve it.
  Block *best = NULL, *prev = NULL;
  CUMEM_CHECK(find_best_block(best, prev, size));
  if( !best )
    return CUMEM_STATUS_OUT_OF_MEMORY;
  CUMEM_CHECK(extract_block(best, prev, size, true));
  block_data = best->get_data();
  block_size = best->get_size();
  
  // Release the memory used by that block.
  delete best;
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t Manager::print_list(FILE *file, const char *name, const Block *head) const
{
  std::size_t size = 0;
  for( Block *curr = (Block*) head; curr; curr = curr->get_next() )
    size += curr->get_size();
  fprintf(file, "| list=\"%s\", size=%lu\n", name, size);
  for( Block *curr = (Block*) head ; curr ; curr = curr->get_next() )
  {
    fprintf(file, "| | node=0x%016lx, data=0x%016lx, size=%lu, next=0x%016lx, head=%2lu\n", 
      (std::size_t) curr, 
      (std::size_t) curr->get_data(),
      (std::size_t) curr->get_size(),
      (std::size_t) curr->get_next(),
      (std::size_t) curr->is_head ());
  }
  fprintf(file, "|\n");
  return size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::release(void *ptr)
{
  // Skip if ptr is NULL.
  if( ptr == NULL )
  {
    printf("release(NULL)\n");
    CUMEM_STATUS_SUCCESS;
  }
    
  // Lock to make sure only one thread execute that fragment of code.
  Lock lock;
  CUMEM_CHECK(lock.lock(&m_mutex));
  
  // Find the node in the list of used blocks.
  Block *curr = m_used_blocks, *prev = NULL;
  for( ; curr && curr->get_data() != ptr ; curr = curr->get_next() )
    prev = curr;
  
  // Make sure we have found a node.
  if( curr == NULL )
  {
    CUMEM_CHECK(lock.unlock());
    return CUMEM_STATUS_INVALID_ARGUMENT;
  }

  // We have the node so release it.
  cumemStatus_t result = release_block(curr, prev);
  CUMEM_CHECK(lock.unlock());
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::release_all(bool &memory_leaks)
{
  // Destroy the children if any.
  bool ok = true;
  for( std::size_t i = 0; i < m_children.size(); ++i )
  {
    bool tmp;
    CUMEM_CHECK(m_children[i].release_all(tmp));
    ok = ok && !tmp;
  }
  // TODO: thejaswi: HACK! HACK! HACK!
  // We have some issues when integrating into some libraries. This has to fixed in the libs.
  // memory_leaks = !ok || m_used_blocks;
  memory_leaks = !ok;

  // Destroy used blocks. It's a kind of panic mode to avoid leaks. NOTE: Do that only with roots!!!
  if( !m_parent )
    while( m_used_blocks )
      CUMEM_CHECK(release_block(m_used_blocks, NULL));

  // We should be having only free blocks that are head blocks. Release those blocks.
  while( m_free_blocks )
  {
    if( m_parent )
      CUMEM_CHECK(m_parent->release(m_free_blocks->get_data()));
    else if( m_free_blocks->is_head() )
    {
      #ifdef CUMEM_DEBUG
      std::cout << "attempting cudaFree of size " << m_free_blocks->get_size()
                << "B address=" << (void*)m_free_blocks->get_data() << std::endl;
      #endif
      CHECK_CUDA(cudaFree(m_free_blocks->get_data()));
      #ifdef CUMEM_DEBUG
      std::cout << "cudaFree of size " << m_free_blocks->get_size() << "B successful" << std::endl;
      #endif
    }
    Block *block = m_free_blocks;
    m_free_blocks = m_free_blocks->get_next();
    delete block;
  }

  // We shouldn't have any used block left. Or, it means the user is causing memory leaks!
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::release_block(Block *curr, Block *prev)
{
  // The current node cannot be NULL!
  assert(curr != NULL);
  
  // Change the connection of the node.
  if( prev )
    prev->set_next(curr->get_next());
  else
    m_used_blocks = curr->get_next();
    
  // Find the location where this block should be added to the free list.
  prev = NULL;
  Block *iter = m_free_blocks;
  for( ; iter && iter->get_data() < curr->get_data() ; iter = iter->get_next() )
    prev = iter;
  
  // Keep track of the successor of pred. We may lose track of it in the following "else".
  Block *next = prev ? prev->get_next() : m_free_blocks;
  
  // We first check if we can merge the block with its predecessor in the list and curr can be merged.
  if( prev && prev->get_data() + prev->get_size() == curr->get_data() && !curr->is_head() )
  {
    prev->set_size(prev->get_size() + curr->get_size());
    delete curr;
    curr = prev;
  }
  else if( prev )
    prev->set_next(curr);
  else
    m_free_blocks = curr;
  
  // Check if we can merge curr and next. We can't merge over "cudaMalloc" boundaries.
  if( next && curr->get_data() + curr->get_size() == next->get_data() && !next->is_head() )
  {
    curr->set_size(curr->get_size() + next->get_size());
    curr->set_next(next->get_next());
    delete next;
  }
  else
    curr->set_next(next);
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::reserve(std::size_t size)
{
  Block *curr, *prev;
  CUMEM_CHECK(allocate_block(curr, prev, size));
  m_size = size;
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::steal(void *&stolen, std::size_t size)
{
  // If we cannot steal, don't even try.
  if( m_flags & CUMEM_FLAGS_CANNOT_STEAL )
  {
    stolen = NULL;
    return CUMEM_STATUS_INVALID_ARGUMENT;
  }

  // The stolen block.
  void *data = NULL; std::size_t data_size = 0;
  if( !m_children.empty() )
    CUMEM_CHECK(steal_block(data, data_size, size));
  else if( m_parent )
    CUMEM_CHECK(m_parent->steal_block(data, data_size, size));
  
  // Make sure we do have a block of memory or quit.
  if( !data )
  {
    stolen = NULL;
    return CUMEM_STATUS_OUT_OF_MEMORY;
  }

  // Push the block in the used list.
  m_used_blocks = new Block((char*) data, data_size, m_used_blocks, true);

  // Return the new pointer into memory.
  stolen = data;
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t Manager::steal_block(void *&data, std::size_t &data_size, ::size_t size)
{
  // No block found and no room to grow. Try to steal from a children (if we have any).
  data = NULL;
  for( std::size_t i = 0 ; !data && i < m_children.size() ; ++i )
    if( m_children[i].give_block(data, data_size, size) == CUMEM_STATUS_SUCCESS )
      break;
    
  // If no memory space found, simply return NULL. We have failed to allocate. Quit miserably.
  if( !data )
    return CUMEM_STATUS_OUT_OF_MEMORY;

  // We have got a node from a children. We need to update our "used" list before we can do 
  // anything with it.
  Block *curr = m_used_blocks, *prev = NULL;
  for( ; curr ; curr = curr->get_next() )
  {
    if( curr->get_data() <= data && data < curr->get_data()+curr->get_size() )
      break;
    prev = curr;
  }
  
  // Curr points to the node which contains that memory region.
  assert(curr);

  // If it is exactly the same memory region, we are done!!!
  if( curr->get_data() == data && curr->get_size() == data_size )
    return CUMEM_STATUS_SUCCESS;
  
  // Track the blocks before and after curr.
  Block *next = curr->get_next();
  
  // We may have up to 3 blocks.
  std::size_t size_before = (std::size_t) ((char*) data - curr->get_data());
  std::size_t size_after  = (curr->get_size() - size_before - data_size);

  // The resulting block.
  Block *result = curr;
  
  // If we have no space between curr->get_data and block->get_data.
  if( size_before == 0 )
    curr->set_size(data_size);
  else
  {
    curr->set_size(size_before);
    Block *block = new Block((char*) data, data_size, next, false);
    curr->set_next(block);
    curr = block;
    data = (char*) data + data_size;
    data_size = size_after; 
    result = block;
  }
  
  // We have space at the end so we may need to add a new node.
  if( size_after > 0 )
  {
    Block *block = new Block(curr->get_data() + curr->get_size(), size_after, next, false);
    curr->set_next(block);
    curr = block;
  }

  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static void print_blocks(FILE *file, const Manager &manager)
{
  fprintf(file, "device=%d, stream=0x%016lx, used=%luB, free=%luB\n", 
      manager.get_device(), 
      (std::size_t) manager.get_stream(),
      manager.get_used_memory(),
      manager.get_free_memory());
  manager.print_used_blocks(file);
  manager.print_free_blocks(file);
  fprintf(file, "\n");
  
  for( std::size_t i = 0 ; i < manager.get_children().size() ; ++i )
    print_blocks(file, manager.get_children()[i]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cumem

///////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t cumemInit(int numDevices, const cumemDevice_t *devices, unsigned flags)
{
  // Make sure we have at least one device declared.
  CHECK(numDevices > 0, CUMEM_STATUS_INVALID_ARGUMENT);
  
  // Find the largest ID of the device.
  int max_device = 0;
  for( int i = 0 ; i < numDevices ; ++i )
    if( devices[i].device > max_device )
      max_device = devices[i].device;
    
  // Allocate enough managers.
  CHECK(max_device >= 0, CUMEM_STATUS_INVALID_ARGUMENT);
  std::vector<cumem::Manager> &managers = cumem::Manager::get_root_managers();
  managers.resize(max_device+1);

  // Create a root manager for each device and create the children.
  for( int i = 0 ; i < numDevices ; ++i )
  {
    std::size_t size = devices[i].size;
    if( size == 0 )
    {
      cudaDeviceProp props;
      CHECK_CUDA(cudaGetDeviceProperties(&props, devices[i].device));
      size = props.totalGlobalMem / 2;
    }
    CHECK(size > 0, CUMEM_STATUS_INVALID_ARGUMENT);

    std::size_t granularity = devices[i].granularity;
    if( granularity == 0 )
    {
      granularity = CUMEM_DEFAULT_GRANULARITY;
    }
    CHECK(granularity > 0, CUMEM_STATUS_INVALID_ARGUMENT);
    CHECK(((granularity % 512) == 0), CUMEM_STATUS_INVALID_ARGUMENT);
    
    cumem::Manager &manager = cumem::Manager::get_root_manager(devices[i].device);
    manager.set_device(devices[i].device);
    manager.set_flags(flags);
    manager.set_granularity(granularity);
    
    size = ((size + granularity - 1) / granularity) * granularity;
    CUMEM_CHECK(manager.reserve(size));
    
    std::vector<cumem::Manager> &children = manager.get_children();
    children.resize(devices[i].numStreams);
    
    for( int j = 0 ; j < devices[i].numStreams ; ++j )
    {
      children[j].set_parent(&manager);
      children[j].set_device(devices[i].device);
      children[j].set_stream(devices[i].streams[j]);
      children[j].set_flags(flags & ~CUMEM_FLAGS_CANNOT_GROW);
      CUMEM_CHECK(children[j].reserve(size / (devices[i].numStreams + 1)));
    }
  }
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t cumemFinalize()
{
  if( cumem::Manager::get_root_managers().empty() )
    return CUMEM_STATUS_NOT_INITIALIZED;
    
  std::vector<cumem::Manager> &managers = cumem::Manager::get_root_managers();
  bool memory_leaks = false;
  for( std::size_t i = 0; i < managers.size(); ++i )
  {
    bool tmp_leaks;
    CUMEM_CHECK(managers[i].release_all(tmp_leaks));
    memory_leaks = memory_leaks || tmp_leaks;
  }
  managers.clear();
  return memory_leaks ? CUMEM_STATUS_MEMORY_LEAK : CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t cumemMalloc(void **ptr, std::size_t size, cudaStream_t stream)
{
  if( cumem::Manager::get_root_managers().empty() )
    return CUMEM_STATUS_NOT_INITIALIZED;
  if( !ptr && !size )
    return CUMEM_STATUS_SUCCESS;
  if( !ptr )
    return CUMEM_STATUS_INVALID_ARGUMENT;
  if( !size )
    return CUMEM_STATUS_INVALID_ARGUMENT;
  
  int device;
  CHECK_CUDA(cudaGetDevice(&device));

  cumem::Manager &root = cumem::Manager::get_root_manager(device);
  cumem::Manager *manager = &root;
  if( stream )
    CUMEM_CHECK(root.get_child(manager, stream));
  assert(manager);

  size_t granularity = manager->get_granularity();
  size = ((size + granularity - 1) / granularity) * granularity;
  cumemStatus_t result = manager->allocate(ptr[0], size);

  // We failed to allocate but there might still be a buffer available in another manager. Try to 
  // steal it.
  if( result == CUMEM_STATUS_OUT_OF_MEMORY )
  {
    // We need to acquire all the locks to all the managers to be able to steal memory. It's costly!
    typedef std::vector<cumem::Manager>::iterator Iterator;


    // Try to acquire locks on all the children.
    std::vector<cumem::Manager> &children = root.get_children();
    std::vector<cumem::Lock> locks(children.size() + 1);
    std::size_t num_locked = 0;
    for( Iterator it = children.begin() ; it != children.end() ; ++it, ++num_locked )
    {
      cumem::Mutex *mutex = it->get_mutex();
      if( locks[num_locked].lock(mutex) != CUMEM_STATUS_SUCCESS )
        break;
    }

    // We locked all the children, so we try to lock the root.
    if( num_locked == children.size() )
    {
      cumemStatus_t tmp_status = locks.back().lock(root.get_mutex());
      if( tmp_status == CUMEM_STATUS_SUCCESS )
        num_locked++;
    }

    // We acquired so we try to steal a node from another child.
    if( num_locked == locks.size() )
      result = manager->steal(ptr[0], size);
    for( std::size_t i = 0 ; i < num_locked ; ++i )
      CUMEM_CHECK(locks[i].unlock());
  }
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t cumemFree(void *ptr, cudaStream_t stream)
{
  if( cumem::Manager::get_root_managers().empty() )
    return CUMEM_STATUS_NOT_INITIALIZED;
  if( ptr == NULL )
    return CUMEM_STATUS_SUCCESS;

  int device;
  CHECK_CUDA(cudaGetDevice(&device));

  cumem::Manager &root = cumem::Manager::get_root_manager(device);
  cumem::Manager *manager = &root;
  if( stream )
    CUMEM_CHECK(root.get_child(manager, stream));
  assert(manager);
  return manager->release(ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t cumemGetMemoryUsage(size_t *used_mem, size_t *free_mem)
{
  if( cumem::Manager::get_root_managers().empty() )
    return CUMEM_STATUS_NOT_INITIALIZED;
  if( !used_mem || !free_mem )
    return CUMEM_STATUS_INVALID_ARGUMENT;
  for( std::size_t i = 0, j = 0 ; i < cumem::Manager::get_root_managers().size() ; ++i )
  {
    cumem::Manager &manager = cumem::Manager::get_root_managers()[i];
    if( manager.get_device() == -1 )
      continue;
    cumem::Lock lock;
    CUMEM_CHECK(lock.lock(manager.get_mutex()));
    used_mem[j] = manager.get_used_memory();
    free_mem[j] = manager.get_free_memory();
    j++;
    CUMEM_CHECK(lock.unlock());
  }
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cumemStatus_t cumemPrintMemoryState(FILE *file)
{
  if( cumem::Manager::get_root_managers().empty() )
    return CUMEM_STATUS_NOT_INITIALIZED;
  for( std::size_t i = 0 ; i < cumem::Manager::get_root_managers().size() ; ++i )
  {
    cumem::Manager &manager = cumem::Manager::get_root_managers()[i];
    if( manager.get_device() == -1 )
      continue;
    cumem::Lock lock;
    CUMEM_CHECK(lock.lock(manager.get_mutex()));
    print_blocks(file, manager);
    CUMEM_CHECK(lock.unlock());
  }
  return CUMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

const char* cumemGetErrorString(cumemStatus_t status)
{
  switch(status)
  {
    case CUMEM_STATUS_SUCCESS: return "CUMEM_STATUS_SUCCESS";
    case CUMEM_STATUS_CUDA_ERROR: return "CUMEM_STATUS_CUDA_ERROR";
    case CUMEM_STATUS_INVALID_ARGUMENT: return "CUMEM_STATUS_INVALID_ARGUMENT";
    case CUMEM_STATUS_MEMORY_LEAK: return "CUMEM_STATUS_MEMORY_LEAK";
    case CUMEM_STATUS_NOT_INITIALIZED: return "CUMEM_STATUS_NOT_INITIALIZED";
    case CUMEM_STATUS_OUT_OF_MEMORY: return "CUMEM_STATUS_OUT_OF_MEMORY";
    default: return "CUMEM_STATUS_UNKNOWN_ERROR";
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // extern "C"

