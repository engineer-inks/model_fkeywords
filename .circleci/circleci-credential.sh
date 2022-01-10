#!/bin/bash

cat <<EOF > ~/.pypirc
[distutils]
index-servers = local
[local]
repository: https://dnaink.jfrog.io/artifactory/api/pypi/dna-ink-pypi
username: $ARTIFACTORY_USERNAME
password: $ARTIFACTORY_PASSWORD
EOF

echo "Created .pypirc file: Here it is: "
ls -la ~/.pypirc