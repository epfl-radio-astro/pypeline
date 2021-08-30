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
                sh 'echo $BUILD_TIMESTAMP'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_lofar_toothbrush_ps.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_fastsynthesizer.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_synthesizer.sh'
                sh 'ls -rtl'
            }
        }
    }
}
