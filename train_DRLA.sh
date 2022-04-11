# lab1
Ns=(10)
# Ns=(10 70)
gpu_id=0
# Ns=(110)
# gpu_id=1

# lab2
# Ns=(30 50)
# gpu_id=0
# Ns=(90)
# gpu_id=1

for N in ${Ns[*]}
do
    echo "now train ${train_policy}"
    nohup python3 -u train_DRLA.py --N $N --gpu_id $gpu_id > ./out/train_${N}.out 2>&1 &
    sleep 3
    
done