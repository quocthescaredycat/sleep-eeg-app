#!/bin/sh
# Runs before nginx starts (official image, /docker-entrypoint.d/). Turns on HTTP Basic Auth
# when both BASIC_AUTH_USER and BASIC_AUTH_PASSWORD are set; refuses to start with only one.
set -eu

dir=/etc/nginx/auth
user=${BASIC_AUTH_USER:-}
password=${BASIC_AUTH_PASSWORD:-}

if [ -z "$user" ] && [ -z "$password" ]; then
    : > "$dir/basic.conf"
    echo "$0: basic auth off"
    exit 0
fi
if [ -z "$user" ] || [ -z "$password" ]; then
    echo "$0: set both BASIC_AUTH_USER and BASIC_AUTH_PASSWORD, or neither" >&2
    exit 1
fi
case $user in *:*)
    echo "$0: BASIC_AUTH_USER must not contain ':'" >&2
    exit 1
    ;;
esac

# SHA-512 crypt; the password never appears on a command line.
hash=$(printf '%s' "$password" | cryptpw -m sha512)
umask 027
printf '%s:%s\n' "$user" "$hash" > "$dir/htpasswd"
chgrp nginx "$dir/htpasswd"
printf 'auth_basic "Sleep staging";\nauth_basic_user_file %s/htpasswd;\n' "$dir" > "$dir/basic.conf"
echo "$0: basic auth on for user $user"
