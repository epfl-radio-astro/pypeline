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

        stage('Testing') {
            steps {                
                sh 'pwd'
                echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                sh "echo Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                sh "echo TAG_DATE = ${env.TAG_DATE}"
                script {
                    def now = new Date()
                    println now.format("yyMMdd.HHmm", TimeZone.getTimeZone('UTC'))
                }

                sh 'echo TAG_UNIXTIME = ${env.TAG_UNIXTIME}'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_lofar_toothbrush_ps.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_fastsynthesizer.sh'
                //sh 'srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G ./jenkins/slurm_test_synthesizer.sh'
                //sh 'ls -rtl'
            }
        }
    }
}
