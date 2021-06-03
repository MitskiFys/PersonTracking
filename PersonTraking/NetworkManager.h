#ifndef NETWORKMANAGER_H
#define NETWORKMANAGER_H

#include "HelpEntities.cpp"
#include <QtCore/QObject>
#include <QNetworkAccessManager>
#include "opencv2/core/ocl.hpp"

class NetworkManager : public QObject {
    Q_OBJECT
public:
    explicit NetworkManager(pt::QueueFPS<cv::Mat>& frameSourse, QObject* parent = nullptr);
    ~NetworkManager();
    void makeGetRequest();
    void setAccessToken(const QString& accessToken);
    void setHttpPort(int port);
private slots:
    void onFinished(QNetworkReply* reply);

private:
    QString m_accessToken;
    int m_httpPort;
    QNetworkAccessManager m_manager;
    pt::QueueFPS<cv::Mat>* frameSourse;

};

#endif // NETWORKMANAGER_H