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
                sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test.sh'
                sh 'ls -rtl'
                //sh 'sbatch ./jenkins/slurm_test.sh'
            }
        }
    }
}
