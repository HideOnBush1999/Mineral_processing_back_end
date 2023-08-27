import bcrypt

# 加密并返回哈希值和使用的盐，这个的输出为 b'XXXXXXXXXXX' 表示是字节序列，实际上返回值类型都为bytes
# 保存到数据库的时候，可以用string类型只保存 XXXXXXXXXXX，读取出数据后，再使用 .encode()变成 bytes类型
def encrypt_and_salt_password(password):
    # 生成随机盐并使用 bcrypt 加密密码
    salt = bcrypt.gensalt(rounds=12)
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password, salt


# 验证密码是否匹配
def verify_password(input_password, hashed_password, salt):
    hashed_input_password = bcrypt.hashpw(input_password.encode('utf-8'), salt)
    return hashed_password == hashed_input_password


if __name__ == "__main__":
    # 示例使用
    plain_password = "mysecretpassword"

    # 加密密码，并获取哈希和盐
    hashed_password, salt = encrypt_and_salt_password(plain_password)
    print(type(hashed_password))
    print(type(salt))

    print("Hashed password:", hashed_password)
    print("Salt:", salt)

    # 模拟用户登录验证
    user_input_password = "mysecretpassword"
    if verify_password(user_input_password, hashed_password, salt):
        print("Password is correct.")
    else:
        print("Password is incorrect.")

    plain_password = "123456"
    hashed_password, salt = encrypt_and_salt_password(plain_password)
    print("Hashed password:", hashed_password)
    print("Salt:", salt)

    plain_password = "111111"
    hashed_password, salt = encrypt_and_salt_password(plain_password)
    print("Hashed password:", hashed_password)
    print("Salt:", salt)
