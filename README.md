### 后端（Flask）命名规范

1. **变量命名：** 同样使用驼峰命名法，但是首字母小写。例如：`userData`, `requestParameters`。
2. **函数命名：** 使用下划线分隔的小写字母来命名函数，以动词开头描述函数的操作。例如：`get_user_data()`, `update_user_profile()`。
3. **模块命名：** 使用下划线分隔的小写字母来命名模块文件。例如：`user_management.py`, `data_processing.py`。
4. **路由命名：** 使用下划线分隔的小写字母来命名路由，描述其功能。例如：`/get_user_data`, `/update_profile`。
5. **数据库表命名：** 使用下划线分隔的小写字母，表名通常使用复数形式。例如：`users`, `products`。
6. **常量命名：** 全部大写，多个单词间用下划线分隔。例如：`MAX_ATTEMPTS`, `DEFAULT_TIMEOUT`。
7. **数据库字段命名：** 使用下划线分隔的小写字母，保持与表名和实际数据结构的一致性。例如：`first_name`, `created_at`。
8. **文件夹命名：** 文件夹命名可以遵循类似模块文件的规则，使用下划线分隔的小写字母来命名文件夹，描述其包含内容的特点。例如：`user_management`, `data_processing`。
9. **数据库指令命名：** 关键字都使用大写。例如：`SELECT password, salt FROM users WHERE username = %s`



### 数据库

用户名: root

密码: 123456

数据库名称： mine

选定的字符集为: utf8mb4

选定的排序规则为: utf8mb4_general_ci



创建数据库

```SQL
CREATE DATABASE mine;

USE mine;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(255) NOT NULL,
    salt VARCHAR(100) NOT NULL,
);

INSERT INTO users (username, password, salt) VALUES
('admin', '$2b$12$3c2Nkuon.78tQMhY1eLhWujVLIfPYZpRb0kU/9LDSPpjFuG4g2VDa', '$2b$12$3c2Nkuon.78tQMhY1eLhWu'),
('cheng', '$2b$12$r6.pPWaHOLONR3bxrRGFLu30jxsbdRYnP/bpHovjkLqjIK3.XnxBC', '$2b$12$r6.pPWaHOLONR3bxrRGFLu');
```

minio 对象存储

单例   资源管理

Websocket 加 流式推送
流式推送（如逐字逐句地发送大语言模型的回答）最适合使用 WebSocket。这是因为 WebSocket 提供了全双工通信，允许服务器实时地向客户端发送数据，而不需要客户端不断地发起新的请求。

Celery 异步任务队列  -->  用在模型的训练耗时任务上  和 go 语言中的 Asynq 库类似，Asynq 使用 Redis 作为消息代理
Celery 是一个基于 Python 的异步任务队列，它可以轻松地将耗时的任务异步化，并通过消息队列（如 RabbitMQ、Redis 等）将任务结果返回给客户端。
