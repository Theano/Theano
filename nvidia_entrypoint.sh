#!/bin/bash
set -e
cat <<EOF
                                                                                                                                                
============
== Theano ==
============

NVIDIA Release ${NVIDIA_THEANO_VERSION} (build ${NVIDIA_BUILD_ID})

Container image Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
Copyright (c) 2008--2016, Theano Development Team
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.
EOF

if [[ "$(find /usr -name libcuda.so.1) " == " " || "$(ls /dev/nvidiactl) " == " " ]]; then
  echo
  echo "WARNING: NVIDIA Driver not detected.  GPU functionality will not be available."
  echo "   Please use 'nvidia-docker run' to invoke this container; see /workspace/README.md for details."
fi

echo

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
