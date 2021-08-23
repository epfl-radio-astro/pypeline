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
                sh 'ls -rlth'
            }
        }
    }
}
