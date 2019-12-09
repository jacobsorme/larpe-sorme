for i in $(seq 1 1 20)
do
        python3 cartpole_dqn.py --file freq --freq $i
done
