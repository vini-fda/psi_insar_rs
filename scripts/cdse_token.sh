#!/bin/bash

set -eEuo pipefail

# This is a script to login to the CDSE and get the access token
# Reference: https://documentation.dataspace.copernicus.eu/APIs/Token.html

read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
echo
read -s -p "2FA Token: " TOTP
echo

if [ -z "$TOTP" ]; then
    echo "Error: You have no more attempts left to enter the 2FA token."
    exit 1
fi

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Error: Username, password and 2FA token required"
    exit 1
fi

export CDSE_ACCESS_TOKEN=$(curl -d 'client_id=cdse-public' \
                    -d "username=$USERNAME" \
                    -d "password=$PASSWORD" \
                    -d 'grant_type=password' \
                    -d "totp=$TOTP" \
                    'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' | jq -r .access_token)
