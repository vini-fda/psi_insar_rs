#!/bin/zsh

# This is a script to login to the CDSE and get the access token
# To run this script and properly export the token, you need to source it:
# source scripts/cdse_token.sh
# Reference: https://documentation.dataspace.copernicus.eu/APIs/Token.html

read "CDSE_USERNAME?username: "
read -s "CDSE_PASSWORD?password: "
echo
read -s "TOTP?2FA token: "
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

USERNAME_ENCODED=$(urlencode "$CDSE_USERNAME")
PASSWORD_ENCODED=$(urlencode "$CDSE_PASSWORD")

if [ -z "$TOTP" ]; then
    echo "Error: You have no more attempts left to enter the 2FA token."
    #exit 1
fi

if [ -z "$CDSE_USERNAME" ] || [ -z "$CDSE_PASSWORD" ]; then
    echo "Error: Username, password and 2FA token required"
    #exit 1
fi

RESPONSE=$(curl -s -d 'client_id=cdse-public' \
                    -d "username=$USERNAME_ENCODED" \
                    -d "password=$PASSWORD_ENCODED" \
                    -d 'grant_type=password' \
                    -d "totp=$TOTP" \
                    'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token')

ACCESS_TOKEN=$(echo "$RESPONSE" | jq -r .access_token)

if [ "$ACCESS_TOKEN" != "null" ] && [ -n "$ACCESS_TOKEN" ]; then
    echo "Success! Token obtained."
    
    export CDSE_ACCESS_TOKEN="$ACCESS_TOKEN"
else
    echo "Error: Failed to obtain access token"
    echo "Response: $RESPONSE"
fi
