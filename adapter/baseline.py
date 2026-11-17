"""ACLNN V3 dropless routing plus full alignment and output adapters."""
import ctypes as C
import torch

P=C.c_void_p; I=C.c_int64; U=C.c_uint64
acl=C.CDLL('libnnopbase.so',mode=C.RTLD_GLOBAL)
api=C.CDLL('libopapi.so',mode=C.RTLD_GLOBAL)
acl.aclCreateTensor.restype=P
acl.aclCreateTensor.argtypes=[C.POINTER(I),U,C.c_int,C.POINTER(I),I,C.c_int,C.POINTER(I),U,P]
acl.aclDestroyTensor.argtypes=[P]; acl.aclCreateIntArray.restype=P
acl.aclCreateIntArray.argtypes=[C.POINTER(I),U]; acl.aclDestroyIntArray.argtypes=[P]
api.aclnnMoeInitRoutingV3GetWorkspaceSize.restype=C.c_int
api.aclnnMoeInitRoutingV3GetWorkspaceSize.argtypes=[P,P,P,P,I,I,I,I,I,C.c_bool,I,P,I,P,P,P,P,C.POINTER(U),C.POINTER(P)]
api.aclnnMoeInitRoutingV3.restype=C.c_int
api.aclnnMoeInitRoutingV3.argtypes=[P,U,P,P]
DT={torch.float32:0,torch.int32:3,torch.int64:9,torch.uint8:4}

def tensor_handle(t):
 shape=(I*t.ndim)(*t.shape); stride=(I*t.ndim)(*t.stride())
 h=acl.aclCreateTensor(shape,t.ndim,DT[t.dtype],stride,0,2,shape,t.ndim,P(t.data_ptr()))
 if not h: raise RuntimeError('aclCreateTensor failed')
 return h

def routing(ids,e,h=1):
 t,k=ids.shape; n=t*k
 x=torch.zeros((t,h),dtype=torch.float32,device=ids.device)
 xx=torch.empty((n,h),dtype=torch.float32,device=ids.device)
 rows=torch.empty(n,dtype=torch.int32,device=ids.device)
 counts=torch.empty(e,dtype=torch.int64,device=ids.device)
 scale=torch.empty(0,dtype=torch.float32,device=ids.device)
 tensors=[tensor_handle(v) for v in (x,ids,xx,rows,counts,scale)]
 bounds=(I*2)(0,e); arr=acl.aclCreateIntArray(bounds,2); size=U(); executor=P()
 if not arr: raise RuntimeError('aclCreateIntArray failed')
 try:
  xh,ih,xxh,rh,ch,sh=tensors
  rc=api.aclnnMoeInitRoutingV3GetWorkspaceSize(xh,ih,None,None,n,0,e,0,1,True,-1,arr,1,xxh,rh,ch,sh,C.byref(size),C.byref(executor))
  if rc: raise RuntimeError(f'V3 workspace failed: {rc}')
  workspace=torch.empty(size.value,dtype=torch.uint8,device=ids.device)
  rc=api.aclnnMoeInitRoutingV3(P(workspace.data_ptr()),size,executor,P(torch.npu.current_stream().npu_stream))
  if rc: raise RuntimeError(f'V3 execute failed: {rc}')
  return rows,counts
 finally:
  for a in tensors: acl.aclDestroyTensor(a)
  acl.aclDestroyIntArray(arr)

def full(ids,e,h=1):
 t,k=ids.shape; n=t*k; cap=n+(e+1)*63; blocks=(cap+63)//64
 row,counts=routing(ids,e,h)
 ends=torch.cumsum(counts,0); padded=torch.div(counts+63,64,rounding_mode='floor')*64; pend=torch.cumsum(padded,0)
 zero=torch.zeros(1,dtype=torch.int64,device=ids.device)
 start=torch.cat((zero,ends[:-1])); pstart=torch.cat((zero,pend[:-1]))
 route=row.to(torch.int64); expert=ids.reshape(-1)[route].to(torch.int64)
 dst=pstart[expert]+torch.arange(n,device=ids.device,dtype=torch.int64)-start[expert]
 out=torch.full((cap,),n,dtype=torch.int32,device=ids.device); out.scatter_(0,dst,row)
 pos=torch.arange(blocks,device=ids.device,dtype=torch.int64)*64
 owner=torch.searchsorted(pend.contiguous(),pos,right=True).to(torch.int32)
 owner=torch.where(owner<e,owner,-1)
 return out,owner,pend[-1:].to(torch.int32)
