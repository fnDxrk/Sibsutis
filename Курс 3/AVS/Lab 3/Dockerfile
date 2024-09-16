FROM ubuntu:latest

RUN apt update && apt install -y openssh-server python3 python3-pip sudo

RUN mkdir /var/run/sshd

RUN echo 'root:2024' | chpasswd

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
