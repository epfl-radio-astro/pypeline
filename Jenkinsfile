pipeline {
    agent {
        label 'izar-orliac'
    }

    stages {

        stage('Build') {
            steps {
                sh 'conda create --name=pypeline --channel=defaults --channel=conda-forge --file=conda_requirements.txt'
                sh 'source pypeline.sh --no_shell'
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
