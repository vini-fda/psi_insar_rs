#!/bin/bash

set -eEuo pipefail

# This is a script to login to the CDSE and get the access token
# Reference: https://documentation.dataspace.copernicus.eu/APIs/Token.html

read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
echo
read -s -p "2FA Token: " TOTP
echo

urlencode() {
    # Function to URL encode the username and password
    local string="${1}"
    local strlen=${#string}
    local encoded=""
    local pos c o

    for (( pos=0 ; pos<strlen ; pos++ )); do
        c=${string:$pos:1}
        case "$c" in
            [-_.~a-zA-Z0-9] ) o="${c}" ;;
            * ) printf -v o '%%%02x' "'$c" ;;
        esac
        encoded+="${o}"
    done
    echo "${encoded}"
}

USERNAME_ENCODED=$(urlencode "$USERNAME")
PASSWORD_ENCODED=$(urlencode "$PASSWORD")

if [ -z "$TOTP" ]; then
    echo "Error: You have no more attempts left to enter the 2FA token."
    exit 1
fi

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Error: Username, password and 2FA token required"
    exit 1
fi

export CDSE_ACCESS_TOKEN=$(curl -d 'client_id=cdse-public' \
                    -d "username=$USERNAME_ENCODED" \
                    -d "password=$PASSWORD_ENCODED" \
                    -d 'grant_type=password' \
                    -d "totp=$TOTP" \
                    'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' | jq -r .access_token)
