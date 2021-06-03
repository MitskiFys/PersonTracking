#ifndef ECHOCLIENT_H
#define ECHOCLIENT_H

#include <QtCore/QObject>
#include <QtWebSockets/QWebSocket>

class EchoClient : public QObject
{
    Q_OBJECT
public:
    explicit EchoClient(QWebSocket &websocket, bool debug = false, QObject* parent = nullptr);
    QString getAccesToken();
    int getHttpPort();
    bool isGetAccesToken();
    bool isGetHttpPort();

    void ptzLeft();
    void ptzRight();
    void ptzStop();
    void ptzUp();
    void ptzDown();
Q_SIGNALS:
    void closed();

public Q_SLOTS:
    void onConnected();
    void onTextMessageReceived(QString message);

private:
    std::atomic<bool> m_isGetAccesToken;
    std::atomic<bool> m_isGetHttpPort;
    QString m_accesToken;
    int m_httpPort;
    QWebSocket *m_webSocket;
    bool m_debug;
};

#endif // ECHOCLIENT_H