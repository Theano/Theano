node('docker-theano') {
    stage('Theano flake8 doctest') {
        checkout scm
        sh 'git clean -fdx'
        sh ".jenkins/jenkins_pretest.sh"
        junit '*tests.xml'
    }
}

parallel firstBranch: {
        node('docker-theano') {
            stage('Theano core') {
                checkout scm
                sh 'git clean -fdx'
                sh ".jenkins/jenkins_test1.sh"
                junit '*tests.xml'
            }
        }
    }, secondBranch: {
        node('docker-cuda-theano') {
            stage('Theano GPU') {
                checkout scm
                sh 'git clean -fdx'
                sh ".jenkins/jenkins_test2.sh"
                junit '*tests.xml'
            }
        }
    },
    failFast: false|false
