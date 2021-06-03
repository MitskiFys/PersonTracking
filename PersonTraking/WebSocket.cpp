#include "WebSocket.h"
#include <iostream>
#include <QtCore/QDebug>
#include "qthread.h"
#include <QJsonDocument>
#include <QJsonObject>
QT_USE_NAMESPACE

//! [constructor]
EchoClient::EchoClient(QWebSocket& websocket, bool debug, QObject* parent) :
    QObject(parent),
    m_webSocket(&websocket),
    m_isGetAccesToken(false),
    m_httpPort(0),
    m_isGetHttpPort(false),
    m_debug(debug)
{
    if (m_debug)
        qDebug() << "WebSocket server:";
    connect(m_webSocket, &QWebSocket::connected, this, &EchoClient::onConnected);
    connect(m_webSocket, &QWebSocket::disconnected, this, &EchoClient::closed);
    connect(m_webSocket, &QWebSocket::textMessageReceived, this, &EchoClient::onTextMessageReceived, Qt::AutoConnection);
    connect(m_webSocket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error),
        [=](QAbstractSocket::SocketError error) {
            qDebug() << "[IoBoard] Error: " << error;
        }
    );
}
//! [constructor]

//! [onConnected]
void EchoClient::onConnected()
{
    if (m_debug)
        qDebug() << "WebSocket connected";
    std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"setUsedApiVersion\",\"version\" : \"1\"}")) << std::endl;
    QThread::msleep(1);
}
//! [onConnected]

QString EchoClient::getAccesToken()
{
    return m_accesToken;
}

int EchoClient::getHttpPort()
{
    return m_httpPort;
}

bool EchoClient::isGetAccesToken()
{
    return m_isGetAccesToken.load();
}

bool EchoClient::isGetHttpPort()
{
    return m_isGetHttpPort.load();
}

void EchoClient::ptzLeft()
{
    std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"ptzLeft\"}")) << std::endl;
    QThread::msleep(1);
}

void EchoClient::ptzRight()
{
    std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"ptzRight\"}")) << std::endl;
    QThread::msleep(1);
}

void EchoClient::ptzStop()
{
    std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"ptzStop\"}")) << std::endl;
    QThread::msleep(1);
}

void EchoClient::ptzUp()
{
    std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"ptzUp\"}")) << std::endl;
    QThread::msleep(1);
}

void EchoClient::ptzDown()
{
    std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"ptzDown\"}")) << std::endl;
    QThread::msleep(1);
}

//! [onTextMessageReceived]
void EchoClient::onTextMessageReceived(QString message)
{
    const auto state = m_webSocket->state();
    if (m_debug)
        qDebug() << "Message received:" << message;

    
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    QJsonObject sett2 = doc.object();
    const auto val = sett2.value(QString("method")).toString();

    if (val == "setUsedApiVersion")
    {
        std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"auth\",\"type\" : \"secured\", \"credentials\" : \"123\"}")) << std::endl;
        QThread::msleep(1);
    }
    else if (val == "auth")
    {
        m_accesToken = sett2.value(QString("tokenForHttpServer")).toString();
        m_isGetAccesToken.store(true);
        std::cout << m_webSocket->sendTextMessage(QString("{\"method\" : \"getHttpServerSettings\"}")) << std::endl;
        QThread::msleep(1);
    }
    else if (val == "getHttpServerSettings")
    {
        auto settings = sett2.value(QString("settings"));
        m_httpPort = settings[0]["value"].toInt();
        m_isGetHttpPort.store(true);
    }
}
//! [onTextMessageReceived]