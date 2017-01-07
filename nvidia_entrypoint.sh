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

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
