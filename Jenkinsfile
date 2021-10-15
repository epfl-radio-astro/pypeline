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

        stage('Standard CPU') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/test_standard_cpu"
            }

            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
            }
        }

        stage('Standard GPU') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/test_standard_gpu"
                TEST_ARCH = '--gpu'
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
            }
        }

        stage('lofar_bootes_nufft_small_fov') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/lofar_bootes_nufft_small_fov"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_lofar_bootes_nufft_small_fov.sh"
            }
        }

        stage('Monitoring') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/monitoring"
                TEST_FSTAT = "${env.OUT_DIR}/monitoring/stats.txt"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_monitoring.sh"
                sh "cat ${env.TEST_FSTAT}"
                script {
                    def data = readFile("${env.TEST_FSTAT}")
                    if (data.contains("_WARNING_")) {
                        println("_WARNING_ found\n");
                        slackSend color:'warning', message:"_WARNING(s)_ detected in stats!\n${data}\n${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    } else {
                        println("_WARNING_ NOT found...\n");
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
