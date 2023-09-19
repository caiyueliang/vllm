FROM swr.cn-central-221.ovaijisuan.com/wair/pytorch_2_0_0:pytorch_2.0.0-cuda_11.8.0-py_3.9-ubuntu_20.04

USER root

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install pip -U -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install -r /tmp/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# COPY ./opt-125m/ /tmp/opt-125m/
# RUN pip install ray pandas -i https://mirrors.aliyun.com/pypi/simple/

COPY . /home/vllm/
RUN pip install pip -U -i https://mirrors.aliyun.com/pypi/simple/ \
    && cd /home/vllm/ \
    && pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /home/server

# CMD ["python", "src/server.py"]
CMD ["bash", "run.sh"]
