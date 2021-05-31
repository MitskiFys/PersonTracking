#include "WebSocket.h"
#include <QtCore/QDebug>

QT_USE_NAMESPACE

//! [constructor]
EchoClient::EchoClient(QWebSocket& websocket, bool debug, QObject* parent) :
    QObject(parent),
    m_webSocket(&websocket),
    m_debug(debug)
{
    if (m_debug)
        qDebug() << "WebSocket server:";
    connect(m_webSocket, &QWebSocket::connected, this, &EchoClient::onConnected);
    connect(m_webSocket, &QWebSocket::disconnected, this, &EchoClient::closed);
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
    connect(m_webSocket, &QWebSocket::textMessageReceived,
        this, &EchoClient::onTextMessageReceived);
    m_webSocket->sendTextMessage(QString("{\"method\" : \"setUsedApiVersion\",\"version\" : \"1\"}"));
}
//! [onConnected]

//! [onTextMessageReceived]
void EchoClient::onTextMessageReceived(QString message)
{
    if (m_debug)
        qDebug() << "Message received:" << message;
    m_webSocket->close();
}
//! [onTextMessageReceived]