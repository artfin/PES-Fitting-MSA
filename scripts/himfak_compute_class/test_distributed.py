import os                                                                                                                                                                                                 
import socket                                                                                                                                                                                             
import torch
import torch.distributed as dist                                                                                                                                                                          
                
def main():
    dist.init_process_group(backend="nccl")
                                                                                                                                                                                                          
    rank = dist.get_rank()
    world_size = dist.get_world_size()                                                                                                                                                                    
    hostname = socket.gethostname()

    torch.cuda.set_device(0)                                                                                                                                                                              
    device = torch.device("cuda:0")
                                                                                                                                                                                                          
    print(f"[Rank {rank}/{world_size}] {hostname}: {torch.cuda.get_device_name()}")                                                                                                                       

    # Test AllReduce                                                                                                                                                                                      
    tensor = torch.ones(1000, device=device) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)                                                                                                                                                         
    expected_sum = sum(range(world_size))                                                                                                                                                                 

    if torch.allclose(tensor, torch.full_like(tensor, expected_sum)):                                                                                                                                     
        print(f"[Rank {rank}] AllReduce PASSED")
    else:                                                                                                                                                                                                 
        print(f"[Rank {rank}] AllReduce FAILED")
                                                                                                                                                                                                          
    dist.barrier()
    if rank == 0:
        print("\n=== All nodes connected! ===")
                                                                                                                                                                                                          
    dist.destroy_process_group()                                                                                                                                                                          
                                                                                                                                                                                                          
if __name__ == "__main__":                                                                                                                                                                                
    main()      

