pipeline {

    agent {
        label 'izar-orliac'
    }

    environment {
        UTC_TAG  = "${sh(script:'date -u +"%Y-%m-%dT%H-%M-%SZ"', returnStdout: true).trim()}"
        WORK_DIR = "/work/scitas-share/SKA/jenkins/izar-orliac"
        OUT_DIR  = "${env.WORK_DIR}/${env.GIT_BRANCH}/${env.UTC_TAG}_${env.BUILD_ID}"
    }

    stages {

        stage('Build') {
            steps {
                sh 'echo !! install.sh disabled !!'
                //sh 'sh ./jenkins/install.sh'
            }
        }

        stage('Standard CPU') {

            environment {
                TEST_DIR  = "${env.OUT_DIR}/test_synthesizer_cpu"
            }

            steps {
                sh 'pwd'
                sh 'env'
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 4 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_test_synthesizer_cpu.sh"
            }
        }

        stage('Standard GPU') {

            environment {
                TEST_DIR  = "${env.OUT_DIR}/test_synthesizer"
            }

            steps {
                sh 'pwd'
                sh 'env'
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 4 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_test_synthesizer.sh"
           }
        }
    }
}
