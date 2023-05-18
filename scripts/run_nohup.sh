m=${1}
n=${2}
p=${3}
g=${4}

nohup python train.py -m ${m} -n ${n} -p ${p} -g ${g} > /dev/null 2>&1 &
