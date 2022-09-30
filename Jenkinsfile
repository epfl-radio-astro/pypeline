def AGENT_LABEL = "izar-ska"
def QOS = "gpu_free"

pipeline {

    agent {
        label "${AGENT_LABEL}"
    }

    environment {
        //UTC_TAG  = "${sh(script:'date -u +"%Y-%m-%dT%H-%M-%SZ"', returnStdout: true).trim()}"
        UTC_TAG  = "${sh(script:'module load git; TZ=UTC0 git show -s --quiet --date=format-local:\'%Y-%m-%dT%H-%M-%SZ\' | grep Date | sed -E -e \'s/Date:\\s+//\'', returnStdout: true).trim()}"
        WORK_DIR = "/work/backup/ska/ci-jenkins/${AGENT_LABEL}"
        REF_DIR  = "/work/backup/ska/ci-jenkins/references"
        OUT_DIR  = "${env.WORK_DIR}/${env.GIT_BRANCH}/${env.UTC_TAG}_${env.BUILD_ID}"

        // Set to "1" to run corresponding profiling
        //
        PROFILE_CPROFILE = "0"
        PROFILE_NSIGHT   = "0"
        PROFILE_VTUNE    = "0"
        PROFILE_ADVISOR  = "0" // can be very time-consuming

        // To compile C++ port of bluebild
        //
        NINJA_ROOT = "${env.WORKSPACE}/ninja"
        PATH = "${env.PATH}:${env.NINJA_ROOT}"
        FINUFFT_ROOT    = "${env.WORKSPACE}/finufft"
        CUFINUFFT_ROOT  = "${env.WORKSPACE}/cufinufft"
        BB_SH_LIB       = "${env.WORKSPACE}/jenkins/bluebild.sh"
        LD_LIBRARY_PATH = "${env.LD_LIBRARY_PATH}:${env.FINUFFT_ROOT}/lib:${env.CUFINUFFT_ROOT}/lib"
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
                TEST_DIR = "${env.OUT_DIR}/install"
            }
            steps {
                //slackSend color: 'good', message:"Build Started - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"  
                sh "mkdir -pv ${env.TEST_DIR}"

                //EO: with --full it deletes and rebuild all dependencies
                sh "srun --partition build --time 00-01:00:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_install.sh --full"

                sh "cat ${env.TEST_DIR}/slurm-*.out"
            }
        }


        stage('Doc') {
            environment {
                DOC_DIR  = "${env.OUT_DIR}/doc"
            }
            steps {
                sh "mkdir -pv ${env.DOC_DIR}"
                //sh 'sh ./jenkins/build_documentation.sh'                
            }
        }


        /*
            LOFAR BOOTES NUFFT3
        */

        stage('lofar_bootes_nufft3_cpu_64') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_nufft3_cpu_64"
                CUPY_PYFFS = "0"
                SLURM_OPTS = "--qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out"
                COMMAND    = "${BB_SH_LIB} ${env.WORKSPACE}/jenkins/new_lofar_bootes_nufft3.py \
                              --processing_unit none --precision double --outdir ${env.TEST_DIR}"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: "sbatch --wait --parsable --partition build --time 00-00:15:00 \
                                 ${SLURM_OPTS} ./jenkins/slurm_timing ${COMMAND}",
                        returnStdout: true
                    ).trim()
                    sh "seff ${JOBID} >> ${env.TEST_DIR}/slurm-${JOBID}.out"
                }
                sh "srun --partition debug --time 00-00:30:00 \
                    ${SLURM_OPTS} ./jenkins/slurm_profiling ${COMMAND}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
            }
        }


        stage('lofar_bootes_nufft3_cpp_cpu_64') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_nufft3_cpp_cpu_64"
                CUPY_PYFFS = "0"
                SLURM_OPTS = "--qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out"
                COMMAND    = "${BB_SH_LIB} ${env.WORKSPACE}/jenkins/new_lofar_bootes_nufft3.py \
                              --processing_unit cpu --precision double --outdir ${env.TEST_DIR}"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: "sbatch --wait --parsable --partition build --time 00-00:15:00 \
                                 ${SLURM_OPTS} ./jenkins/slurm_timing ${COMMAND}",
                        returnStdout: true
                    ).trim()
                    sh "seff ${JOBID} >> ${env.TEST_DIR}/slurm-${JOBID}.out"
                }
                sh "srun --partition debug --time 00-00:30:00 \
                    ${SLURM_OPTS} ./jenkins/slurm_profiling ${COMMAND}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
            }
        }


        stage('lofar_bootes_nufft3_cpp_gpu_64') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_nufft3_cpp_gpu_64"
                CUPY_PYFFS = "0"
                SLURM_OPTS = "--qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out"
                COMMAND    = "${BB_SH_LIB} ${env.WORKSPACE}/jenkins/new_lofar_bootes_nufft3.py \
                              --processing_unit gpu --precision double --outdir ${env.TEST_DIR}"
                PROFILE_NSIGHT = "0"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: "sbatch --wait --parsable --partition build --time 00-00:15:00 \
                                 ${SLURM_OPTS} ./jenkins/slurm_timing ${COMMAND}",
                        returnStdout: true
                    ).trim()
                    sh "seff ${JOBID} >> ${env.TEST_DIR}/slurm-${JOBID}.out"
                }
                sh "srun --partition debug --time 00-00:30:00 \
                    ${SLURM_OPTS} ./jenkins/slurm_profiling ${COMMAND}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
            }
        }


        /*
            LOFAR BOOTES SS
        */

        stage('lofar_bootes_ss_cpu_64') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_ss_cpu_64"
                CUPY_PYFFS = "0"
                SLURM_OPTS = "--qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out"
                COMMAND    = "${BB_SH_LIB} ${env.WORKSPACE}/jenkins/new_lofar_bootes_ss.py \
                              --processing_unit none --precision double --outdir ${env.TEST_DIR}"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: "sbatch --wait --parsable --partition build --time 00-00:15:00 \
                                 ${SLURM_OPTS} ./jenkins/slurm_timing ${COMMAND}",
                        returnStdout: true
                    ).trim()
                    sh "seff ${JOBID} >> ${env.TEST_DIR}/slurm-${JOBID}.out"
                }
                sh "srun --partition debug --time 00-00:30:00 \
                    ${SLURM_OPTS} ./jenkins/slurm_profiling ${COMMAND}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
            }
        }

        stage('lofar_bootes_ss_cpp_cpu_64') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_ss_cpp_cpu_64"
                CUPY_PYFFS = "0"
                SLURM_OPTS = "--qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out"
                COMMAND    = "${BB_SH_LIB} ${env.WORKSPACE}/jenkins/new_lofar_bootes_ss.py \
                              --processing_unit cpu --precision double --outdir ${env.TEST_DIR}"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: "sbatch --wait --parsable --partition build --time 00-00:15:00 \
                                 ${SLURM_OPTS} ./jenkins/slurm_timing ${COMMAND}",
                        returnStdout: true
                    ).trim()
                    sh "seff ${JOBID} >> ${env.TEST_DIR}/slurm-${JOBID}.out"
                }
                sh "srun --partition debug --time 00-00:30:00 \
                    ${SLURM_OPTS} ./jenkins/slurm_profiling ${COMMAND}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
            }
        }


        stage('lofar_bootes_ss_cpp_gpu_64') {
            environment {
                TEST_DIR   = "${env.OUT_DIR}/lofar_bootes_ss_cpp_gpu_64"
                CUPY_PYFFS = "0"
                SLURM_OPTS = "--qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 20 -o ${env.TEST_DIR}/slurm-%j.out"
                COMMAND    = "${BB_SH_LIB} ${env.WORKSPACE}/jenkins/new_lofar_bootes_ss.py \
                              --processing_unit gpu --precision double --outdir ${env.TEST_DIR}"
                PROFILE_NSIGHT = "0"
            }
            steps {
                sh "mkdir -pv ${env.TEST_DIR}"
                script {
                    JOBID = sh (
                        script: "sbatch --wait --parsable --partition build --time 00-00:15:00 \
                                 ${SLURM_OPTS} ./jenkins/slurm_timing ${COMMAND}",
                        returnStdout: true
                    ).trim()
                    sh "seff ${JOBID} >> ${env.TEST_DIR}/slurm-${JOBID}.out"
                }
                sh "srun --partition debug --time 00-00:30:00 \
                    ${SLURM_OPTS} ./jenkins/slurm_profiling ${COMMAND}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
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
                sh "srun --partition build --time 00-00:15:00 --qos ${QOS} --gres gpu:1 --mem 40G --cpus-per-task 1 \
                    -o ${env.TEST_DIR}/slurm-%j.out ./jenkins/slurm_monitoring.sh ${BB_SH_LIB}"
                sh "cat ${env.TEST_DIR}/slurm-*.out"
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
