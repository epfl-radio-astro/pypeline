#!/bin/bash

set -e

# 29/11/2021: Jenkins history was lost after renaming of the workspace
#             so it started again from build no 1
# => renaming the current solution directory
mv -v /work/backup/ska/ci-jenkins/izar-ska/eo_jenkins/  /work/backup/ska/ci-jenkins/izar-ska/eo_jenkins_old
