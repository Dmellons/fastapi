from main import get_password_hash


if __name__ == '__main__':
    # username = input('Enter username: ')
    password = input('Enter password: ')

    pwd = get_password_hash(password)
    print(pwd)