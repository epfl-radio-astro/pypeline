def AGENT_LABEL = "izar-ska"

pipeline {

    agent {
        label "${AGENT_LABEL}"
    }

    environment {
        UTC_TAG  = "${sh(script:'date -u +"%Y-%m-%dT%H-%M-%SZ"', returnStdout: true).trim()}"
        WORK_DIR = "/work/backup/ska/ci-jenkins/${AGENT_LABEL}"
        REF_DIR  = "/work/backup/ska/ci-jenkins/references"
        OUT_DIR  = "${env.WORK_DIR}/${env.GIT_BRANCH}/${env.UTC_TAG}_${env.BUILD_ID}"
    }

    stages {

        stage('Build') {
            environment {
                OMP_NUM_THREADS = "1"
            }
            steps {
                slackSend color: 'good', message:"Build Started - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                sh 'echo REMINDER: installation (./jenkins/install.sh) disabled'
                //sh 'source ~/.bashrc'
                //sh 'sh ./jenkins/install.sh'
            }
        }

        stage('Seff') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/seff"
                TEST_SEFF = "1"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: 'sbatch --wait --parsable --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh',
                        returnStdout: true
                    ).trim()
                    echo "Seff JOBID: ${JOBID}"
                }
                echo "Seff JOBID (bis): ${JOBID}"
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

        stage('lofar_bootes_nufft3') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_nufft3"
                CUPY_PYFFS = "0"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_lofar_bootes_nufft3.sh"
            }
        }

        stage('Monitoring') {
            environment {
                TEST_DIR       = "${env.OUT_DIR}/monitoring"
                TEST_FSTAT_RT  = "${env.OUT_DIR}/monitoring/stats_rt.txt"
                TEST_FSTAT_IMG = "${env.OUT_DIR}/monitoring/stats_img.txt"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_monitoring.sh"
                sh "cat ${env.TEST_FSTAT_RT}"
                script {
                    def data = readFile("${env.TEST_FSTAT_RT}")
                    if (data.contains("_WARNING_")) {
                        println("_WARNING_ found\n");
                        slackSend color:'warning', message:"_WARNING(s)_ detected in run times statistics!\n${data}\n${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    } else {
                        println("_WARNING_ NOT found...\n");
                    }
                }
                sh "cat ${env.TEST_FSTAT_IMG}"
                script {
                    def data = readFile("${env.TEST_FSTAT_IMG}")
                    if (data.contains("_WARNING_")) {
                        println("_WARNING_ found\n");
                        slackSend color:'warning', message:"_WARNING(s)_ detected in image statistics!\n${data}\n${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
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
