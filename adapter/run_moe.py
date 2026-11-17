"""PR44 full manifest correctness and paired hardware timing, under host guard."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import traceback
from pathlib import Path
from guard import require_reservation

p=argparse.ArgumentParser();p.add_argument('--phase',choices=['accuracy','performance'],required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--case',action='append');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
result={'stage':'starting','cases':[],'phase':a.phase}
def save(): (a.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
try:
 require_reservation()
 import numpy as np
 import torch,torch_npu,tilelang,tileops
 torch.set_num_threads(1);torch.npu.set_device(0)
 torch.npu.set_device_limit(0,cube_num=32,vector_num=64)
 assert torch.npu.get_device_limit(0)=={'cube_core_num':32,'vector_core_num':64}
 from tileops.moe import MoEPermuteAlignFwdOp
 from tileops.manifest import entry_plan,instantiate,load_manifest,load_adts
 from workloads.device import set_run_device
 from benchmarks._cases.moe import ENTRIES
 from benchmarks.api import Case
 from baseline import full
 from hardware_timer import Timer
 set_run_device('npu:0')
 entry=load_manifest()['MoEPermuteAlignFwdOp'];plan=entry_plan('MoEPermuteAlignFwdOp',entry,load_adts(),resolve=False)
 cases=[Case(call.case_id,call,ENTRIES['MoEPermuteAlignFwdOp']) for call in (instantiate(plan,r,d) for r in entry['workloads'] for d in r.get('dtype_cases',[{}]))]
 assert len(cases)==12,len(cases)
 cases=[c for c in cases if a.case is None or c.id in a.case]
 assert cases and (a.case is None or len(cases)==len(a.case))
 props=torch.npu.get_device_properties(0)
 result['identity']={'device':torch.npu.get_device_name(0),'physical':[props.cube_core_num,props.vector_core_num],
   'limits':torch.npu.get_device_limit(0),'tilelang':tilelang.__file__,'tileops':tileops.__file__,
   'versions':{n:importlib.metadata.version(n) for n in ['torch','torch_npu','tilelang','tileops']}}
 assert '20261116-pr17-pr44-mainline' in tilelang.__file__,tilelang.__file__
 assert '/moe/package/' in tileops.__file__,tileops.__file__
 def check(ids,values,e):
  assert len(values)==3
  actual=[x.cpu().numpy() for x in values]
  flat=ids.cpu().numpy().reshape(-1);n=flat.size;cap=n+(e+1)*63
  out,owner,count=actual
  assert out.shape==(cap,) and owner.shape==((cap+63)//64,) and count.shape==(1,)
  assert all(x.dtype==np.int32 for x in actual)
  expected=np.argsort(flat,kind='stable');counts=np.bincount(flat,minlength=e)
  read=write=0
  for expert,num in enumerate(counts):
   pad=(num+63)//64*64
   assert np.array_equal(np.sort(out[write:write+num]),expected[read:read+num]),expert
   assert np.all(out[write+num:write+pad]==n),expert
   assert np.all(owner[write//64:(write+pad)//64]==expert),expert
   read+=num;write+=pad
  assert count[0]==write and np.all(out[write:]==n)
  assert np.all((owner[write//64:]<0)|(owner[write//64:]>=e))
 timer=Timer(a.output/'timer') if a.phase=='performance' else None
 for case in cases:
  torch.manual_seed(1235);torch.npu.manual_seed_all(1235)
  inputs=case.inputs;ids=inputs[0];protected=ids.clone();e=case.params['num_experts']
  row={'case':case.id,'shape':list(ids.shape),'experts':e};result['cases'].append(row)
  result['stage']=case.id;save();print('START',case.id,flush=True)
  op=MoEPermuteAlignFwdOp(e,64)
  for label,fn in [('tilelang',lambda:op(ids)),('aclnn-v3-composed',lambda:full(ids,e,h=0))]:
   values=fn();torch.npu.synchronize();check(ids,values,e)
   assert torch.equal(ids,protected)
   row[label]={'accuracy':'passed','outputs_checked':3,'input_preserved':True}
   if label=='tilelang':
    kernel=list(op.iter_kernels())[0]
    assert type(kernel).__name__=='AscendMoEPermuteAlignKernel'
    source=kernel.kernels.get_kernel_source()
    (a.output/(case.id+'.asc')).write_text(source)
    (a.output/(case.id+'.host.cpp')).write_text(kernel.kernels.get_host_source())
    assert ('__schedmode__(1)' in source)==(kernel.grid>1)
    assert 'ascend.batch_schedule' not in source and 'moe_grid_sync' not in source
    row['grid']=kernel.grid;row['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
     streamed=op(ids);consumed=tuple(x.clone() for x in streamed)
    stream.synchronize();check(ids,consumed,e)
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(values,streamed))
    row[label]['nondefault_stream']='passed'
   if timer:
    row[label]['timing']=timer.measure(fn,a.output/case.id/label)
   del values
   save()
  if timer:
   def sig(label):return collections.Counter(tuple(s['frequency_before']+s['frequency_after']) for s in row[label]['timing']['samples'])
   row['frequency_comparable']=sig('tilelang')==sig('aclnn-v3-composed')
   row['ratio']=row['aclnn-v3-composed']['timing']['mean_device_union_us']/row['tilelang']['timing']['mean_device_union_us']
   print('RATIO',case.id,row['ratio'],'frequency',row['frequency_comparable'],flush=True)
  print('PASS',case.id,flush=True);save()
  del ids,inputs,protected,op
  case._release();torch.npu.empty_cache()
 result['stage']='complete';result['status']='pending_guard_receipt';save()
except BaseException:
 result.update(status='failed',error=traceback.format_exc());save();raise
