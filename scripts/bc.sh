sbatch -p cpu --mem=10000 scripts/job_bc.sh CartPole-v1 ./dataset/cartpole/StableCartPole-v1.npz 1024
sbatch -p cpu --mem=10000 scripts/job_bc.sh Acrobot-v1 ./dataset/acrobot/StableAcrobot-v1.npz 1024
sbatch -p cpu --mem=10000 scripts/job_bc.sh LunarLander-v2 ./dataset/lunarlander/StableLunarLander-v2.npz 1024
sbatch -p cpu --mem=10000 scripts/job_bc.sh MountainCar-v0 ./dataset/mountaincar/StableMountainCar-v0.npz 1024