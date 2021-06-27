#include "NetworkManager.h"
#include <QNetworkReply>
#include <iostream>
#include <QImage>
#include <qthread.h>


NetworkManager::NetworkManager(pt::QueueFPS<cv::Mat>& frameSourse, QObject* parent) :
    QObject(parent)
{
    this->frameSourse = &frameSourse;
    connect(&m_manager, SIGNAL(finished(QNetworkReply*)), SLOT(onFinished(QNetworkReply*)));
}

NetworkManager::~NetworkManager() {
}

void NetworkManager::setAccessToken(const QString& accessToken)
{
    m_accessToken = accessToken;
}

void NetworkManager::setHttpPort(int port)
{
    m_httpPort = port;
}

void NetworkManager::makeGetRequest()
{
    QString url;
    url.append(QString("http://10.110.15.232:")).append(QString::number(m_httpPort)).append("/frames/?peerId=%23self%3A0&token=").append(m_accessToken).append("&hash=0.3969172338125255").trimmed();
    auto reply = m_manager.get(QNetworkRequest(QUrl(url)));
}

cv::Mat QImage2Mat(QImage const& src)
{
    cv::Mat tmp(src.height(), src.width(), CV_8UC4, (uchar*)src.bits(), src.bytesPerLine());
    cv::Mat dst;
    cvtColor(tmp, dst, cv::COLOR_BGRA2BGR);
    return dst;
}

void NetworkManager::onFinished(QNetworkReply* reply) {
    QImage* image = new QImage();
    image->loadFromData(reply->readAll());
    const auto format = image->format();
    //image->rgbSwapped();
    if (image->isNull())
        std::cout << "oops";
    frameSourse->push(QImage2Mat(*image).clone());
    delete image;
    reply->deleteLater();
    QThread::msleep(75);
    makeGetRequest();
}