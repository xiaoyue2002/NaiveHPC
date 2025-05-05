# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset toy --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset arxiv --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset collab --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset citation --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset ddi --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset protein --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset ppa --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset reddit.dgl --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset products --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset youtube --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset amazon_cogdl --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset yelp --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset wikikg2 --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset am --len 32 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset arxiv --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset collab --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset citation --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset ddi --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset protein --len 256 --datadir ~/PA3/data/
srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset ppa --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset reddit.dgl --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset products --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset youtube --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset amazon_cogdl --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset yelp --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset wikikg2 --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset am --len 256 --datadir ~/PA3/data/

# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset collab_reorder --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset arxiv_reorder --len 256 --datadir ~/PA3/data/
# srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset citation_reorder --len 256 --datadir ~/PA3/data/
srun -N 1 --gres=gpu:1 ~/PA3/build/test/unit_tests --dataset ppa_reorder --len 256 --datadir ~/PA3/data/

# # 