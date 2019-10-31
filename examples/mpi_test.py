import os
from mpi4py import MPI   
import tensorflow as tf

if __name__ == "__main__":
    
    # Get MPI rank, we assume one process by node
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Retrieve the list of nodes from SLURM
    hosts_list=os.environ['SLURM_NODELIST']
    print(rank)
    if rank ==0 :
        print(hosts_list)
    hosts_list = ["cgpu"+ s for s in hosts_list.split("cgpu[")[1].split("]")[0].replace('-',',').split(",")]
    
    if rank == 0:
        print(hosts_list)
    
    # Turns it into a list of tasks assuming 8 GPUs per nodes
    mesh_hosts = [hosts_list[i] + ":%d"%(8222+j) for i in range(len(hosts_list)) for j in range(1)]
    
    if rank == 0:
        print(mesh_hosts)
    
    # Create a cluster from the mesh hosts.
    cluster = tf.train.ClusterSpec({"mesh": mesh_hosts})
    
    # Create a server for local mesh members
    server = tf.train.Server(cluster,
                             job_name="mesh",
                             task_index=rank)
    if rank >0:
        server.join()
    else:
        # Let's implement a trivial computatation
        with tf.device("/job:mesh/task:0/device:GPU:0"):
            a1 = tf.random_normal(shape=[128,10000], dtype=tf.float32)

        
        with tf.device("/job:mesh/task:1/device:GPU:0"):
            a2 = tf.random_normal(shape=[128,10000], dtype=tf.float32)
        
        with tf.device("/job:mesh/task:0/device:GPU:1"):
            a3 = a1 + a2
            res = tf.reduce_mean(a3)
        
        with tf.Session(server.target) as sess:
            print(sess.run(res))
            
    