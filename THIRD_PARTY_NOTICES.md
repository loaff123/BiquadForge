# Third-party notices

The optional tests/reference/cmsisdsp-1.10.3 fixture contains unmodified official
Arm CMSIS-DSP sources and their transitively included local headers extracted from
the cmsisdsp Python distribution 1.10.3 source archive. This is a Python package
version, not an inferred CMSIS C release number. Original revision, copyright and
Apache-2.0 notices are retained in every file. The complete upstream LICENSE is
included beside the fixture. No upstream source has been modified.

Copyright (C) 2010-2021 ARM Limited or its affiliates (individual files may state
other ranges). Licensed under Apache License, Version 2.0. See the fixture LICENSE.

Source: https://pypi.org/project/cmsisdsp/1.10.3/
Archive SHA-256: 307d32298faa7c58db912851027fce35f886fec7eb1a27c4b309ce1a97f47519
File hashes and pinned identities are in src/biquadforge/reference_hashes.json.
The normal runtime has no cmsisdsp dependency and never downloads source.

The reject_issue127 example uses numeric coefficients from the public Arm support
thread https://github.com/ARM-software/CMSIS-DSP/issues/127. It is a closed support
question, not evidence of an unfixed CMSIS kernel bug. The other example designs
are original simple fixtures or SciPy-generated Butterworth designs.
