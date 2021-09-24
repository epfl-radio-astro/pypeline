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
                slackSend color: 'good', message:"Build Started - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                sh 'echo !! install.sh disabled !!'
                //sh 'sh ./jenkins/install.sh'
            }
        }

        stage('Parallel Stage') {

            parallel {

                stage('Standard CPU') {

                    environment {
                        TEST_DIR  = "${env.OUT_DIR}/test_standard_cpu"
                    }

                    steps {
                        sh "mkdir -pv ${env.TEST_DIR}"
                        sh "srun --partition gpu --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 4 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
                    }
                }

                stage('Standard GPU') {

                    environment {
                        TEST_DIR  = "${env.OUT_DIR}/test_standard_gpu"
                        TEST_ARCH = '--gpu'
                    }

                    steps {
                        sh "mkdir -pv ${env.TEST_DIR}"
                        sh "srun --partition gpu --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 4 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
                   }
                }
            
               stage('Periodic CPU') {

                    environment {
                        TEST_DIR  = "${env.OUT_DIR}/test_periodic_cpu"
                        TEST_ALGO = '--periodic'
                    }

                    steps {
                        sh "mkdir -pv ${env.TEST_DIR}"
                        sh "srun --partition gpu --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 4 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
                   }
                }
            }
        }
    }
        
    post {
        success {
            slackSend color:'good', message:"Build succeeded  - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
        }
        failure {
            slackSend color:'danger', message:"Build failed  - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
        }
    }
}
