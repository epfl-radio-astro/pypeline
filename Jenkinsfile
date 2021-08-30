pipeline {

    agent {
        label 'izar-orliac'
    }

    stages {

        stage('Build') {
            steps {
                sh 'echo !! install.sh disabled !!'
                //sh 'sh ./jenkins/install.sh'
            }
        }

        environment {
            UTC_TAG = "${sh(script:'date -u +"%Y-%m-%dT%H-%M-%SZ"', returnStdout: true).trim()}"
        }

        stage('Testing') {
            steps {                
                sh 'pwd'
                echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                sh "echo Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                sh "echo UTC_TAG = ${env.UTC_TAG}"
                script {
                    def now = new Date()
                    println now.format("yyyy-MM-dd'T'HH-mm-ss'Z'", TimeZone.getTimeZone('UTC'))
                }

                

                //fails sh 'echo TAG_UNIXTIME = ${env.TAG_UNIXTIME}'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_lofar_toothbrush_ps.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_fastsynthesizer.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_synthesizer.sh'
                //sh 'ls -rtl'
            }
        }
    }
}
