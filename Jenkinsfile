pipeline {

    agent {
        label 'izar-orliac'
    }

    environment {
        WORK_DIR = "/work/scitas-share/SKA/jenkins/izar-orliac"
        UTC_TAG = "${sh(script:'date -u +"%Y-%m-%dT%H-%M-%SZ"', returnStdout: true).trim()}"
    }

    stages {

        stage('Build') {
            steps {
                sh 'echo !! install.sh disabled !!'
                //sh 'sh ./jenkins/install.sh'
            }
        }

        stage('Testing') {
            steps {                
                sh 'pwd'
                echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}, Git branch ${env.GIT_BRANCH}"
                echo "WORK_DIR = ${env.WORK_DIR}"
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_lofar_toothbrush_ps.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_fastsynthesizer.sh'
                sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_synthesizer.sh'
                //sh 'ls -rtl'
            }
        }
    }
}
