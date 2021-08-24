pipeline {
    agent {
        label 'izar-orliac'
    }

    stages {

        stage('Build') {
            steps {
                sh 'sh ./jenkins/install.sh'
            }
        }

        stage('Testing') {
            steps {                
                sh 'pwd'
                sh 'srun -t 00:01:00 --partition=gpu --qos=gpu --gres=gpu:1 nvidia-smi'
                sh 'sbatch ./jenkins/slurm_test.sh'
            }
        }
    }
}
