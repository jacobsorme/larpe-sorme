for i in $(seq 0.5 0.05 0.95)
do
        python3 cartpole_dqn.py --lamb $i --file lamb
done
