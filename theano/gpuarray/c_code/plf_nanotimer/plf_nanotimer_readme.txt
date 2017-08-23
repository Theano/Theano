plf::nanotimer is a ~microsecond-precision cross-platform simple timer class (linux/bsd/mac/windows, C++03/C++11).


Usage is as follows:


	plf::nanotimer timer;


	timer.start()
	// Do something here
	double results = timer.get_elapsed_ns();
	std::cout << "Timing: " << results << " nanoseconds." << std::endl;
	

	timer.start(); // "start" has the same semantics as "restart".
	// Do something else
	results = timer.get_elapsed_ms();
	std::cout << "Timing: " << results << " milliseconds." << std::endl;


	timer.start()
	plf::microsecond_delay(15); // Delay program for 15 microseconds
	results = timer.get_elapsed_us();
	std::cout << "Timing: " << results << " microseconds." << std::endl;




Timer member functions:

void timer.start(): start or restart timer

double timer.get_elapsed_ns(): get elapsed time in nanoseconds

double timer.get_elapsed_us(): get elapsed time in microseconds

double timer.get_elapsed_ms(): get elapsed time in milliseconds



Free-standing functions:

void plf::millisecond_delay(double x): delay the program until x milliseconds have passed

void plf::microseconds_delay(double x): delay the program until x microseconds have passed

void plf::nanoseconds_delay(double x): delay the program until x nanoseconds have passed



I determined that a 'pause'-style function would add too much complexity to the class for simple benchmarking, which in turn might interfere with performance analysis, so if you need a 'pause' function do something like this:

{
	plf::nanotimer timer;


	timer.start()
	// Do something here
	double results = timer.get_elapsed_ns();
	
	// Do something else - timer 'paused'
	
	timer.start()
	
	// Do stuff
	
	results += timer.get_elapsed_ns();
	
	std::cout << "Timing: " << results << " nanoseconds." << std::endl;
}


All plf:: library components are distributed under a Zlib License.
plf::nanotimer (c) Copyright 2016 Matt Bentley
Contact: mattreecebentley@gmail.com
www.plflib.org