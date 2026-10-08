"""Independent integer oracle versus original official C, including injected state."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from oracle import oracle
from biquadforge.fixedpoint import Q15Cascade, _run_scalar
from biquadforge.verify import FLAGS, KERNEL, INIT


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    cc=shutil.which("cc")
    if not cc:pytest.skip("Optional official-C oracle: host compiler unavailable")
    if os.name=="nt":pytest.skip("Test-only ctypes harness currently requires a POSIX shared library")
    root=Path(__file__).parent/"reference"/"cmsisdsp-1.10.3"
    tmp=tmp_path_factory.mktemp("official-c")
    bridge=tmp/"bridge.c"
    bridge.write_text('''#include "dsp/filtering_functions.h"
#include <string.h>
void bridge(unsigned stages, const q15_t *coeff, int post,
 const q15_t *input, q15_t *output, unsigned count,
 const q15_t *initial, q15_t *state, const unsigned *chunks, unsigned nchunks) {
 arm_biquad_casd_df1_inst_q15 instance;
 arm_biquad_cascade_df1_init_q15(&instance,(uint8_t)stages,coeff,state,(int8_t)post);
 memcpy(state,initial,4*stages*sizeof(q15_t));
 unsigned pos=0, step=0;
 while(pos<count) {
  unsigned size=chunks[(step++)%nchunks];
  if(size>count-pos) size=count-pos;
  arm_biquad_cascade_df1_q15(&instance,input+pos,output+pos,size);
  pos+=size;
 }
}
''')
    lib=tmp/"oracle.so"
    command=[cc,*FLAGS,"-shared","-fPIC","-I",str(root/"Include"),"-I",str(root/"PrivateInclude"),str(bridge),str(root/KERNEL),str(root/INIT),"-o",str(lib)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    function=ctypes.CDLL(str(lib)).bridge
    ptr=ctypes.POINTER(ctypes.c_int16);uptr=ctypes.POINTER(ctypes.c_uint)
    function.argtypes=[ctypes.c_uint,ptr,ctypes.c_int,ptr,ptr,ctypes.c_uint,ptr,ptr,uptr,ctypes.c_uint]
    function.restype=None
    def run(co,p,x,initial,schedule):
        output=np.zeros(len(x),dtype=np.int16);state=np.zeros_like(initial)
        chunks=np.array(schedule,dtype=np.uint32)
        function(len(co),co.ctypes.data_as(ptr),p,x.ctypes.data_as(ptr),output.ctypes.data_as(ptr),len(x),initial.ctypes.data_as(ptr),state.ctypes.data_as(ptr),chunks.ctypes.data_as(uptr),len(chunks))
        return output,state
    return run


def test_128_independent_c_cases_exact_outputs_and_state(official):
    rng=np.random.default_rng(731051)
    total=0
    for case in range(128):
        n=[1,2,3,4][case%4];p=(case//4)%4;length=[1,2,3,17,193,257,31,89][case%8]
        co=rng.integers(-32768,32768,(n,6),dtype=np.int16);co[:,1]=0
        if case%7==0:co[:]=32767;co[:,1]=0;co[:,3]=-32768
        x=rng.integers(-32768,32768,length,dtype=np.int16)
        initial=np.zeros((n,4),dtype=np.int16) if case%2==0 else rng.integers(-32768,32768,(n,4),dtype=np.int16)
        expected,expected_state,*_=oracle(co,p,x,initial)
        for schedule in ([1],[3],[7],[16],[127],[length],[1,7,4,31,64]):
            out,state=official(co,p,x,initial,schedule)
            assert out.tolist()==expected,(case,p,n,schedule)
            assert state.tolist()==expected_state,(case,p,n,schedule)
        own,ownstate=_run_scalar(co,p,x,initial)
        assert own.tolist()==expected and ownstate.tolist()==expected_state
        if not initial.any():assert Q15Cascade(co,p).process(x).tolist()==expected
        total+=length
    assert total==9488
