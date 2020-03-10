from sklearn import metrics

if __name__ == '__main__':
    y = [0,0,0,1,1,1]
    y_hat = [0,0,1,1,2,2]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print(u'同一性(homogeneity):', h)
    print(u'完整性(completeness):', c)
    v2 = 2*c*h / (c+h)
    v = metrics.v_measure_score(y, y_hat)
    print(u'V-measure: ', v2, v)

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 3, 3, 3]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print(u'同一性(homogeneity):', h)
    print(u'完整性(completeness):', c)
    v2 = 2 * c * h / (c + h)
    v = metrics.v_measure_score(y, y_hat)
    print(u'V-measure: ', v2, v)

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [1, 1, 1, 0, 0, 0]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print(u'同一性(homogeneity):', h)
    print(u'完整性(completeness):', c)
    v2 = 2 * c * h / (c + h)
    v = metrics.v_measure_score(y, y_hat)
    print(u'V-measure: ', v2, v)