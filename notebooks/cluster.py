from mpi4py import MPI
import mesh_tensorflow as mtf
import tensorflow as tf
import os

FLAGS = None

def main(_):

  # Get MPI rank, we assume one process by node
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Retrieve the list of nodes from SLURM
  # Parse them with ridiculous logic

  mesh_hosts = ["localhost:%d"%(8222+j) for j in range(4)]

  if rank ==0 :
      print(mesh_hosts)

  # Create a cluster from the mesh hosts.
  cluster = tf.train.ClusterSpec({"mesh": mesh_hosts, "master":["localhost:8488"]})

  # Create a server for local mesh members
  server = tf.train.Server(cluster,
                           job_name="mesh",
                           task_index=rank)

  # Only he master job takes care of the graph building
  server.join()


if __name__ == "__main__":
  tf.app.run(main=main)
