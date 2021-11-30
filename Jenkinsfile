def AGENT_LABEL = "izar-ska"
def QOS = "gpu_free"

pipeline {

    agent {
        label "${AGENT_LABEL}"
    }

    environment {
        UTC_TAG  = "${sh(script:'date -u +"%Y-%m-%dT%H-%M-%SZ"', returnStdout: true).trim()}"
        WORK_DIR = "/work/backup/ska/ci-jenkins/${AGENT_LABEL}"
        REF_DIR  = "/work/backup/ska/ci-jenkins/references"
        OUT_DIR  = "${env.WORK_DIR}/${env.GIT_BRANCH}/${env.UTC_TAG}_${env.BUILD_ID}"

        // Set to "1" to run corresponding profiling
        PROFILE_CPROFILE = "1"
        PROFILE_NSIGHT   = "1"
        PROFILE_VTUNE    = "1"
        PROFILE_ADVISOR  = "0" // can be very time-consuming
    }

    stages {

        stage('Management') {
            steps {
                // Run the data management script: all actions on /work/backup/ska/ci-jenkins should
                // be handled through this script
                sh 'sh ./jenkins/data_management.sh'
            }
        }

        stage('Build') {
            environment {
                OMP_NUM_THREADS = "1"
            }
            steps {
                //slackSend color: 'good', message:"Build Started - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"

                // Run the installation script (conda + env + non-conda deps + ref sol)
                // 
                sh 'echo REMINDER: installation \\(./jenkins/install.sh\\) disabled'
                //sh 'sh ./jenkins/install.sh'

                // Cleanup of aborted runs
                //
                //sh "rm -rv ${env.WORK_DIR}/${env.GIT_BRANCH}/2021-11-18T14*"
                //sh "rm -rv ${env.WORK_DIR}/${env.GIT_BRANCH}/2021-11-18T15-[0-2]*"
            }
        }

        // vtune hpc-performance needs to run on debug node!

        stage('Standard CPU') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/test_standard_cpu"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition debug --time 00-00:30:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
            }
        }

        stage('Standard GPU') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/test_standard_gpu"
                TEST_ARCH = '--gpu'
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition debug --time 00-00:30:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh"
            }
        }

        stage('lofar_bootes_nufft_small_fov') {
            environment {
                TEST_DIR  = "${env.OUT_DIR}/lofar_bootes_nufft_small_fov"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition debug --time 00-00:30:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_lofar_bootes_nufft_small_fov.sh"
            }
        }

        stage('lofar_bootes_ss') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_ss"
                CUPY_PYFFS = "0"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition debug --time 00-00:30:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_lofar_bootes_ss.sh"
            }
        }

        stage('lofar_bootes_nufft3') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_nufft3"
                CUPY_PYFFS = "0"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition debug --time 00-00:30:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_lofar_bootes_nufft3.sh"
            }
        }

        stage('Seff') {
            environment {
                TEST_DIR = "${env.OUT_DIR}/seff"
                SEFFDIR_SSCPU = "${env.OUT_DIR}/seff/ss-cpu"
                SEFFDIR_SSGPU = "${env.OUT_DIR}/seff/ss-gpu"
                SEFFDIR_LBSS  = "${env.OUT_DIR}/seff/lb-ss"
                SEFFDIR_LBN   = "${env.OUT_DIR}/seff/lb-n"
                SEFFDIR_LBN3  = "${env.OUT_DIR}/seff/lb-n3"
                TEST_SEFF = "1"
                CUPY_PYFFS = "0"
            }
            steps {
                sh "mkdir -pv ${env.SEFFDIR_SSCPU}"
                script {
                    JOBID = sh (
                        script: "TEST_DIR=${env.SEFFDIR_SSCPU} sbatch --wait --parsable --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.SEFFDIR_SSCPU}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh",
                        returnStdout: true
                    ).trim()
                    sh "echo Seff JOBID: ${JOBID}"
                    sh "seff ${JOBID} >> ${env.SEFFDIR_SSCPU}/slurm-${JOBID}.out"
                }

                sh "mkdir -pv ${env.SEFFDIR_SSGPU}"
                script {
                    JOBID = sh (
                        script: "TEST_ARCH=--gpu TEST_DIR=${env.SEFFDIR_SSGPU} sbatch --wait --parsable --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.SEFFDIR_SSGPU}/slurm-%j.out ./jenkins/slurm_generic_synthesizer.sh",
                        returnStdout: true
                    ).trim()
                    sh "echo Seff JOBID: ${JOBID}"
                    sh "seff ${JOBID} >> ${env.SEFFDIR_SSGPU}/slurm-${JOBID}.out"
                }

                sh "mkdir -pv ${env.SEFFDIR_LBSS}"
                script {
                    JOBID = sh (
                        script: "TEST_DIR=${env.SEFFDIR_LBSS} sbatch --wait --parsable --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.SEFFDIR_LBSS}/slurm-%j.out ./jenkins/slurm_lofar_bootes_ss.sh",
                        returnStdout: true
                    ).trim()
                    sh "echo Seff JOBID: ${JOBID}"
                    sh "seff ${JOBID} >> ${env.SEFFDIR_LBSS}/slurm-${JOBID}.out"
                }

                sh "mkdir -pv ${env.SEFFDIR_LBN}"
                script {
                    JOBID = sh (
                        script: "TEST_DIR=${env.SEFFDIR_LBN} sbatch --wait --parsable --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.SEFFDIR_LBN}/slurm-%j.out ./jenkins/slurm_lofar_bootes_nufft_small_fov.sh",
                        returnStdout: true
                    ).trim()
                    sh "echo Seff JOBID: ${JOBID}"
                    sh "seff ${JOBID} >> ${env.SEFFDIR_LBN}/slurm-${JOBID}.out"
                }
 
                sh "mkdir -pv ${env.SEFFDIR_LBN3}"
                script {
                    JOBID = sh (
                        script: "TEST_DIR=${env.SEFFDIR_LBN3} sbatch --wait --parsable --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.SEFFDIR_LBN3}/slurm-%j.out ./jenkins/slurm_lofar_bootes_nufft3.sh",
                        returnStdout: true
                    ).trim()
                    sh "echo Seff JOBID: ${JOBID}"
                    sh "seff ${JOBID} >> ${env.SEFFDIR_LBN3}/slurm-${JOBID}.out"
                }
            }
        }

        stage('Monitoring') {
            environment {
                TEST_DIR       = "${env.OUT_DIR}/monitoring"
                TEST_FSTAT_RT  = "${env.OUT_DIR}/monitoring/stats_rt.txt"
                TEST_FSTAT_IMG = "${env.OUT_DIR}/monitoring/stats_img.txt"
                TEST_IGNORE_UPTO = "0"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                sh "srun --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_monitoring.sh"
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
