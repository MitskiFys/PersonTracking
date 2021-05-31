#ifndef ECHOCLIENT_H
#define ECHOCLIENT_H

#include <QtCore/QObject>
#include <QtWebSockets/QWebSocket>

class EchoClient : public QObject
{
    Q_OBJECT
public:
    explicit EchoClient(QWebSocket &websocket, bool debug = false, QObject* parent = nullptr);

Q_SIGNALS:
    void closed();

public Q_SLOTS:
    void onConnected();
    void onTextMessageReceived(QString message);

private:
    QWebSocket *m_webSocket;
    bool m_debug;
};

#endif // ECHOCLIENT_H