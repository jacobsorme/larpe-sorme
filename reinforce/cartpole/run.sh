for i in $(seq 64 4 80)
do
        python3 cartpole_dqn.py --file units --units $i
done
