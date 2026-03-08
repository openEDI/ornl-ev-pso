FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN mkdir evcs_federate
COPY . ./evcs_federate
WORKDIR ./evcs_federate
RUN pip install -e .
EXPOSE 5683/tcp
CMD ["python", "-m", "evcs_federate.server"]
