# -*- coding: utf-8 -*-
re = None
datetime = None
am_io = None

def run( aJob ):

	dPP = None

	if aJob['sDataSourceFile']:
		dPP = readCache( aJob )
	# end of if			

	if dPP == None:
		global re
		global datetime
		global am_io
		re = aJob['re']
		datetime = aJob['datetime']
		am_io = aJob['am_io']

		aCandles = []
		aDecisions = []

		aMappings = []
		aDecisionsByMappings = None
		if 'bDecisionsApart' in aJob:
			if aJob['bDecisionsApart'] == True:
				aDecisionsByMappings = []

		if aJob['sDataSourceFile'] == None:
			nAlgorithms, sError = readCandlesAndDecisionsFromDB( aJob, aCandles, aDecisions, aMappings, aDecisionsByMappings )
		else:
			nAlgorithms, sError = readCandlesAndDecisionsFromFiles( aJob, aCandles, aDecisions, aMappings, aDecisionsByMappings )
		if nAlgorithms < 1:
			am_io.log( "Failed to read data." )
			return( -1, "Failed to read data. " + sError )

		if len(aCandles) == 0:
			am_io.log( "No candles found. Exiting..." )
			return( -1, "Failed to read data for the chosen instrument" )

		if len(aDecisions) == 0:
			am_io.log( "No decisions in the mapping(s). Exiting..." )
			return(-1, "No decisions were made by the algorithm(s)" )

		am_io.log( "Deleting unnecessary candles..." ) 
		trimCandles( aCandles, aDecisions )

		am_io.log( "Merging candles and decisions..." ) 
		merge( aCandles, aDecisions )

		am_io.log( "Creating Prices & Profit arrays..." ) 
		dPP = createPP( aJob, aCandles, aDecisions, aMappings, aDecisionsByMappings, nAlgorithms )

		if aJob['sDataSourceFile']:
			writeCache( aJob, dPP )
	# end of if
		
	aJob['dPP'] = dPP

	outputPP( aJob )
	return( 0, "Ok" )
#end of def


def trimCandles( aCandles, aDecisions ):

	iMinCandleSt = aDecisions[0]['iCandleSt']
	iMaxCandleEn = aDecisions[0]['iCandleEn']
	nDecisions = len( aDecisions )
	for iDecision in range( 1, nDecisions ):	
		if aDecisions[iDecision]['iCandleSt'] < iMinCandleSt:
			iMinCandleSt = aDecisions[iDecision]['iCandleSt']
		if aDecisions[iDecision]['iCandleEn'] > iMaxCandleEn:
			iMaxCandleEn = aDecisions[iDecision]['iCandleEn']
	# end of for

	#print "iMinCandleSt=" + str(iMinCandleSt) + ", iMaxCandleEn=" + str(iMaxCandleEn) + ", nC=" + str(len(aCandles))
	#print "min time=" + str(aCandles[iMinCandleSt]['dtDateTime']) + ", max time=" + str(aCandles[iMaxCandleEn]['dtDateTime'])

	aCandles[:] = aCandles[iMinCandleSt:iMaxCandleEn+1]
	for iDecision in range( 0, nDecisions ):	
		aDecisions[iDecision]['iCandleSt'] -= iMinCandleSt
		aDecisions[iDecision]['iCandleEn'] -= iMinCandleSt
# end of def


def createPP( aJob, aCandles, aDecisions, aMappings, aDecisionsByMappings, nAlgorithms ):

	dPP = { 'bOk': False }
	
	aiTime = aJob['ar'].array('i')
	adtTime = []
	afActualProfit = aJob['ar'].array('f')
	afPotentialProfit = aJob['ar'].array('f')
	afClose = aJob['ar'].array('f')
	afNotFixedProfit = aJob['ar'].array('f')
	aiTradesOpen = aJob['ar'].array('i')	

	for iC in range( len( aCandles ) ):
		dC = aCandles[iC]
		aiTime.append( dC['iDateTime'] )
		adtTime.append( dC['dtDateTime'] )
		afActualProfit.append( dC['fActualProfit'] )
		afPotentialProfit.append( dC['fActualProfit'] + dC['fNotFixedProfit'] )
		afNotFixedProfit.append( dC['fNotFixedProfit'] )
		afClose.append( dC['fClose'] )
		aiTradesOpen = dC['nTradesOpened']
		
	dPP['bOk'] = True
	dPP['aiTime'] = aiTime
	dPP['adtTime'] = adtTime
	dPP['afActualProfit'] = afActualProfit
	# dPP['afActualProfitSkippedEqual'] = dropEqualValues( afActualProfit )
	dPP['afPotentialProfit'] = afPotentialProfit
	dPP['afNotFixedProfit'] = afNotFixedProfit
	dPP['afClose'] = afClose
	dPP['aiTradesOpen'] = aiTradesOpen
	dPP['aCandles'] = aCandles
	dPP['aDecisions'] = aDecisions
	dPP['aMappings'] = aMappings
	dPP['aDecisionsByMappings'] = aDecisionsByMappings
	dPP['nAlgorithms'] = nAlgorithms
	return( dPP )
# end of def


def merge( aCandles, aDecisions ):

	nC = len( aCandles )
	for iC in range( nC ):
		aCandles[iC]['fNotFixedProfit'] = 0.0
		aCandles[iC]['fFixedProfit'] = 0.0
		aCandles[iC]['fActualProfit'] = 0.0
		aCandles[iC]['nTradesOpened'] = 0
		aCandles[iC]['nTradesInitiated'] = 0

	nD = len( aDecisions )
	for iD in range( nD ):
		dD = aDecisions[iD]
		iSt = dD['iCandleSt']
		iEn = dD['iCandleEn']											  

		#print "iSt=" + str(iSt) + ", iEn=" + str(iEn) + ", nCandles=" + str(len(aCandles))
		dC = aCandles[iSt]
		nTradesInitiated = dC['nTradesInitiated']
		dC['nTradesInitiated'] = nTradesInitiated + 1
		sKey = 'iTradeInitiated' + str( nTradesInitiated )
		dC[sKey] = iD

		for iC in range( iSt, iEn+1 ):
			dC = aCandles[iC]
			fProfit = 0.0
			if dD['iSide'] == 1:
				if iC < iEn:
					fProfit = dC['fClose'] - dD['fPrice']
				else:
					fProfit = dD['fClosePrice'] - dD['fPrice']
			elif dD['iSide'] == 2:
				if iC < iEn:
					fProfit = dD['fPrice'] - dC['fClose']
				else:
					fProfit = dD['fPrice'] - dD['fClosePrice']
			if iC < iEn:
				dC['fNotFixedProfit'] += fProfit
				nTradesOpened = dC['nTradesOpened']
				dC['nTradesOpened'] = nTradesOpened + 1
				sKey = 'iTradeOpened' + str( nTradesOpened )
				dC[sKey] = iD
			else:
				dC['fFixedProfit'] += fProfit
		# end of for
	# end of for
				
	aCandles[0]['fActualProfit'] = aCandles[0]['fFixedProfit']
	for iC in range( 1, nC ):
		aCandles[iC]['fActualProfit'] = aCandles[iC-1]['fActualProfit'] + aCandles[iC]['fFixedProfit']
	# end of for
# end of def


def dropEqualValues( afValues ):

	nValues = len(afValues)
	afOutputValues = [ [0, afValues[0]] ]
	iOutputLastIndex = 0
	for iValue in range( 1, nValues ):
		if afOutputValues[iOutputLastIndex][1] != afValues[iValue]:
			afOutputValues.append( [iValue, afValues[iValue]] )
			iOutputLastIndex += 1

	return afOutputValues
# end of def		

def outputPP( aJob ):

	dPP = aJob['dPP']
	plt = aJob['plt']
	am_io = aJob['am_io']
	am_utils = aJob['am_utils']
	afNotFixedProfit = dPP['afNotFixedProfit']

	# Displaying algorithms list
	sAList = ""
	if 'iDBSessionId' in aJob:
		sAList += "session=" + str( aJob['iDBSessionId'] ) + "<br/>"
	for i in range( dPP['nAlgorithms'] ):
		sAlgorithm = "";
		if len( dPP['aMappings'][i]['sId'] ) > 0:
			sAlgorithm += "[mapping=" + dPP['aMappings'][i]['sId'] + "] "
		sAlgorithm = dPP['aMappings'][i]['sAlgorithmName']
		sParams = dPP['aMappings'][i]['sParams']
		if sParams != None:
			if len(sParams) > 0:
				sAlgorithm += " (" + sParams + ")"
		if i > 0:
			sAList +="<br/>"
		sAList += sAlgorithm
	am_io.report( "<div>" + sAList + "</div>" )

	am_io.log( "Creating profit plots..." )

	am_io.newFigure( aJob, _sFigureName = u"Доходность + незафиксированная доходность" )
	plt.plot( dPP['adtTime'], dPP['afActualProfit'], label=u"Незафиксированная доходность", color=aJob['sActualProfitColor'] )
	plt.plot( dPP['adtTime'], afNotFixedProfit, label=u"Незафиксированная доходность", color=aJob['sPotentialProfitColor'], dashes=aJob['aiPotentialProfitDashes'] )
	#plt.bar( dPP['adtTime'], dPP['afNotFixedProfit'], alpha=0.8, color='g', error_kw=aJob['dErrorConfig'], label='Not Fixed $' )
	plt.xlabel(u"Время")
	plt.ylabel(u"Доходность + незафиксированная доходность (пункты)")
	plt.legend( bbox_to_anchor=( 0.0, 1.05, 1.0, 0.1), loc=3, ncol=2, mode="expand", borderaxespad=0.0 )
	plt.axhline( 0, color="#afafaf" )
	am_io.saveFigure( aJob, _fXSizeMult = 2.0 )

	sDescription = u"График незафиксированной доходности показывает как меняется возможная прибыль/убыток от позиций\
		пока они остаются открытыми. Если линия графика преимущественно находится <b>выше</b> нулевой линии, это означает,\
		что алгоритм чаще правильно входит в сделку, нежели неправильно. Если линия графика слишком далеко отклоняется\
		от нулевой линии <b>вверх</b>, это означает, что алгоритм упускает возможность для закрытия сделки с большей прибылью.\
		Если линия графика преимущественно находится <b>ниже</b> нулевой линии, это означает,\
		что алгоритм чаще выбирает неподходящие моменты для входа в сделку."
	am_io.newFigure( aJob, _sFigureName = u"Незафиксированная доходность" )
	#plt.plot( dPP['adtTime'], dPP['afNotFixedProfit'], label='Potential Profit', color=aJob['sPotentialProfitColor'], dashes=aJob['aiPotentialProfitDashes'] )
	#plt.bar( dPP['adtTime'], dPP['afNotFixedProfit'], alpha=0.8, color='g', error_kw=aJob['dErrorConfig'], label='Not Fixed $' )
	plt.plot( dPP['adtTime'], afNotFixedProfit, label=u"Незафиксированная доходность", color=aJob['sPriceColor'] )
	plt.xlabel(u'Время')
	plt.ylabel(u'Незафиксированная доходнось (пункты)')
	plt.legend( bbox_to_anchor=( 0.0, 1.05, 1.0, 0.1), loc=3, ncol=3, mode="expand", borderaxespad=0.0 )
	plt.axhline( 0, color="#afafaf" )
	fMean = am_utils.calcMean( afNotFixedProfit )
	fStd =  am_utils.calcStd( fMean, afNotFixedProfit )
	sStat = "Mean=%.1f, Std=%.1f" % (fMean, fStd) 
	plt.text( 0.5, 0.8, sStat, horizontalalignment='center', verticalalignment='center', transform = plt.gca().transAxes )
	am_io.saveFigure( aJob, _fXSizeMult = 2.0, _sDescription = sDescription )
#end of def


def readCandlesAndDecisionsFromDB( aJob, aCandles, aDecisions, aMappings, aDecisionsByMappings ):

	sSystemError = "System error"
	sMultipleInstrumentsError = "Multiple instruments are not supported yet."

	db = aJob['db']

	# Finding session to go with...
	sFields = "interval_start_time, interval_end_time"
	sRequest = "SELECT " + sFields + " FROM sessions WHERE id=" + str(aJob['iDBSessionId']) +  " LIMIT 1"
	bStatus = db.request( sRequest )
	if bStatus == False:
		am_io.log( "Error when searching for the session with id=" + str( aJob['iDBSessionId'] ) + ": " + db.getLastError() )
		return -1, sSystemError

	aRow = db.fetchOne()
	if aRow == None:
		am_io.log( "Failed to find the session with id=" + str( aJob['iDBSessionId'] ) )
		return -1, sSystemError

	iSessionStart = int( aRow[0] )
	iSessionEnd = int( aRow[1] )

	# Reading mappings 
	am_io.log( "Reading mappings..." ) 	
	asMappingsIds = []
	aiMappingsAlgorithms = []
	asMappingsParams = []
	aiMappingsInstruments = []

	if len(aJob['sMappingsIds']) > 0: # mappings are given directly (not all the mapping of the session should be used)
		asMappingsIds = aJob['sMappingsIds'].split(',') 
		for iM in range( len(asMappingsIds) ):

			sFields = "id, trader_id, algorithm_id, params, session_id, is_active, mode, instruments, volume"
			sRequest = "SELECT " + sFields + " FROM algorithm_mapping WHERE id=" + asMappingsIds[iM] + " LIMIT 1;"
			bStatus = db.request( sRequest )
			if bStatus == False:
				am_io.log( "Error reading the mapping with id=" + str(asMappingsIds[iM]) + ": " + db.getLastError() )
				return -1, sSystemError

			aRow = db.fetchOne()
			if aRow == None:
				am_io.log( "None returned when reading the mapping with id=" + str(asMappingsIds[iM]) + ": " + db.getLastError() )
				return -1, sSystemError
			aiMappingsAlgorithms.append( int(aRow[2]) )
			asMappingsParams.append( str(aRow[3]) )
			# aiMappingsIntruments.append( int(aRow[7]) )
			if aRow[7].strip().isdigit():
				aiMappingsInstruments.append( int(aRow[7]) )
			else:
				am_io.log( sMultipleInstrumentsError )
				return -1, sMultipleInstrumentsError
	else: # all mappings of the session should be used
		sFields = "id, trader_id, algorithm_id, params, session_id, is_active, mode, instruments, volume"
		sRequest = "SELECT " + sFields + " FROM algorithm_mapping WHERE session_id=" + str(aJob['iDBSessionId']) + ";"
		bStatus = db.request( sRequest )
		if bStatus == False:
			am_io.log( "Error reading mappings with session id=" + str(aJob['iDBSessionId']) + ": " + db.getLastError() )
			return -1, sSystemError
		while( True ):
			aRow = db.fetchOne()
			if aRow == None:
				break
			asMappingsIds.append( str(aRow[0]) )
			aiMappingsAlgorithms.append( int(aRow[2]) )
			asMappingsParams.append( str(aRow[3]) )
			# aiMappingsIntruments.append( int(aRow[7]) )			
			if aRow[7].strip().isdigit():
				aiMappingsInstruments.append( int(aRow[7]) )
			else:
				am_io.log(sMultipleInstrumentsError)
				return -1, sMultipleInstrumentsError

	nAlgorithms = len( asMappingsIds )
	if nAlgorithms == 0:
		am_io.log( "No mappings in the session." )
		return -1, sSystemError

	# Reading candles
	am_io.log( "Reading candles..." ) 
	iInstrumentId = aiMappingsInstruments[0]
	sFields = "id, time, open_val, max_val, min_val, close_val, instrument_id, time_frame, vol"
	sWhere = "time >= " + str(iSessionStart) + " AND time <= " + str(iSessionEnd) + " AND instrument_id = " + str(iInstrumentId)
	sRequest = "SELECT " + sFields + " FROM candles WHERE " + sWhere
	bStatus = db.request( sRequest )
	if bStatus == False:
		am_io.log( "Error reading candles: " + db.getLastError() )
		return -1, sSystemError
	iCounter = 0
	while( True ):
		aRow = db.fetchOne()
		if aRow == None:
			break
		if len( aRow ) < 9:
			continue
		(iDateTime, dtDateTime) = strToDateTime( str(aRow[1]) )		
		aCandles.append( { 'i': iCounter, 'sTicker': "", 'iPeriod': aRow[7], 'dtDateTime': dtDateTime, 'iDateTime': iDateTime, \
			'fOpen':float(aRow[2]), 'fHigh':float(aRow[3]), 'fLow':float(aRow[4]), 'fClose':float(aRow[5]), 'iVolume':float(aRow[8]) } )
	# end of while

	# Reading decisions
	for iM in range( len( asMappingsIds ) ):
		sFields = "id, mapping_id, price, time, count, decision_type_id, level, position_id, detail_info, side"
		sRequest = "SELECT " + sFields + " FROM decisions WHERE mapping_id=" + asMappingsIds[iM]
		bStatus = db.request( sRequest )
		if bStatus == False:
			am_io.log( "Error reading decisions for the mapping with id=" + asMappingsIds[iM] + ": " + db.getLastError() )
			continue
		aDecisionsRead = []
		while( True ):
			aRow = db.fetchOne()
			if aRow == None:
				break
			if len( aRow ) < 10:
				continue
			(iDateTime, dtDateTime) = strToDateTime( str(aRow[3]) )
			fPrice = float( aRow[2] )
			iCount = int( aRow[4] )
			iPositionId = int( aRow[7] )
			iSide = int( aRow[9] )
			aDecisionsRead.append( { 'dtDateTime':dtDateTime, 'iDateTime':iDateTime, 'fPrice':fPrice, \
				'iCount':iCount, 'iPositionId':iPositionId, 'iSide':iSide } )
		# end of while
		filterDecisions( aJob, aCandles, aDecisionsRead, iM )
		aMappings.append( { 'sId':asMappingsIds[iM], 'iAlgorithmId':aiMappingsAlgorithms[iM], 
			'sParams':asMappingsParams[iM] } )
		if aDecisionsByMappings != None:
			aDecisionsByMappings.append( [] )
			aDecisionsByMappings[iM] = aDecisionsRead[:]
		if len( aDecisionsRead ) > 0:		
			aDecisions.extend( aDecisionsRead )
		del aDecisionsRead
	# end of for

	readAlgorithmsNamesFromDB( aJob, aMappings )

	return 	nAlgorithms, "Ok"
# end of def

def readAlgorithmsNamesFromDB( aJob, aMappings ):
	import psycopg2
	connection = None
	try:
		connection = psycopg2.connect( host = aJob['sDB2HostName'], database = aJob['sDB2Name'], 
	    	user = aJob['sDB2User'], password = aJob['sDB2Password'] ) 
		cursor = connection.cursor()

		for i in range( len(aMappings) ):
			aMappings[i]['sAlgorithm'] = "Unnamed"

			sRequest = "SELECT id, name FROM algorithms WHERE id=" + str( aMappings[i]['iAlgorithmId'] )
			cursor.execute( sRequest )
			while( True ):
				aRow = cursor.fetchone()
				if aRow == None:
					break
				aMappings[i]['sAlgorithmName'] = str( aRow[1] )
			# end of while
		# end of for
	except psycopg2.DatabaseError, e:
		log( "Error %s" % e )    	
	finally:
		if connection:
			connection.close()
# end of def


def readCandlesAndDecisionsFromFiles( aJob, aCandles, aDecisions, aMappings, aDecisionsByMappings ):

	sSystemError = "System error"

	am_io.log( "Reading candles..." ) 
	iStatus = readCandlesFromFile( aJob, aCandles )
	if iStatus < 1:
		return -1, sSystemError
			
	nFiles = len( aJob['aFilesWithDecisions'] )
	for iFile in range( nFiles ):
		am_io.log( "Reading and filtering decisions..." + " (src:" + aJob['aFilesWithDecisions'][iFile] + ")" ) 
		aDecisionsRead = []
		readDecisionsFromFile( aJob, iFile, aDecisionsRead )
		filterDecisions( aJob, aCandles, aDecisionsRead, iFile )
		aMappings.append( { 'sId':aJob['aFilesWithDecisions'][iFile], 'iAlgorithmId':None, 
			'sAlgorithmName':aJob['aFilesWithDecisions'][iFile], 'sParams':None } )
		if aDecisionsByMappings != None:
			aDecisionsByMappings.append( [] )
			aDecisionsByMappings[iFile] = aDecisionsRead[:]
		if len( aDecisionsRead ) > 0:
			aDecisions.extend( aDecisionsRead )
		del aDecisionsRead
	# end of for
	
	return nFiles
# end of def


def dtToInt( dtDateTime ):
	dtOrigin = datetime.datetime( 1970,1,1,0,0,0 )
	iDateTime = (dtDateTime - dtOrigin).total_seconds()
	iDateTime = int( iDateTime )
	return iDateTime
# end of def

# Reads candle data from a file
def readCandlesFromFile( aJob, aCandles ):

	sFile = aJob['sFileWithCandles']
	bOpened = False
	
	nLinesRead = 0
	nLinesSkipped = 0

	try:
		fFile = open(sFile, "r")
		bOpened = True

		for sCandle in fFile:

			reLine = re.match( r'([a-zA-Z0-9\.]+)\,([0-9]+),([0-9]+),([0-9]+),([0-9\.]+),([0-9\.]+),([0-9\.]+),([0-9\.]+),([0-9]+)', sCandle, re.M|re.I)
			
			if reLine:
				sDate = reLine.group(3)
				sTime = reLine.group(4)
				
				reTime = re.match( r'([0-9][0-9])([0-9][0-9])([0-9][0-9])', sTime, re.M|re.I )
				if reTime:
					sH = reTime.group(1)
					sM = reTime.group(2)
					sS = reTime.group(3)
					iH = int (sH )
					iM = int (sM)
					iS = int (sS)
				else:
					nLinesSkipped += 1
					continue
				
				reDate = re.match( r'([0-9][0-9][0-9][0-9])([0-9][0-9])([0-9][0-9])', sDate, re.M|re.I )
				if reDate:
					sYear = reDate.group(1)
					sMonth = reDate.group(2)
					sDay = reDate.group(3)
					iYear = int ( sYear )
					iMonth = int ( sMonth )
					iDay = int ( sDay )
				else:
					nLinesSkipped += 1
					continue
				
				dtDateTime = datetime.datetime( iYear, iMonth, iDay, iH, iM, iS )
				td3Hours = datetime.timedelta(hours=3)
				dtDateTime = dtDateTime - td3Hours
				
				iDateTime = dtToInt( dtDateTime )
			   
				aCandles.append( { 'i': nLinesRead, 'sTicker': reLine.group(1), 'iPeriod': int(reLine.group(2)), \
								  'dtDateTime': dtDateTime, 'iDateTime': iDateTime, \
								  'fOpen':float(reLine.group(5)), 'fHigh':float(reLine.group(6)), \
								  'fLow':float(reLine.group(7)), 'fClose':float(reLine.group(8)), \
								  'iVolume':int(reLine.group(9)) } )
				nLinesRead += 1
			else:
				nLinesSkipped  += 1
				
	except IOError:
		am_io.log(  "Error: can\'t find file " + sFile + " or read data" )
	else:
		am_io.log( "Lines: read=" + str(nLinesRead) + ", skipped=" + str(nLinesSkipped) )
	
	if bOpened:
		fFile.close()

	return nLinesRead
# end of def 
   
	
# Reads decisions from a file
def readDecisionsFromFile( aJob, iFile, aDecisions ):

	sFile = aJob['aFilesWithDecisions'][iFile]
	bOpened = False
	
	nLinesRead = 0
	nLinesSkipped  = 0

	try:
		fFile = open(sFile, "r")
		bOpened = True

		for sDecision in fFile:

			reLine = re.match( r'([0-9]+),([0-9\.]+),([0-9]+),([0-9]+),([0-9]+)', sDecision, re.M|re.I)
		
			if reLine:
				sDateTime = reLine.group(1)
				(iDateTime, dtDateTime) = strToDateTime( sDateTime )
				if iDateTime == None:
					nLinesSkipped += 1
					continue

				aDecisions.append( { 'dtDateTime':dtDateTime, 'iDateTime':iDateTime, \
									'fPrice':float(reLine.group(2)), 'iCount':int(reLine.group(3)), \
									'iPositionId':int(reLine.group(4)), 'iSide':int(reLine.group(5)) } )
				nLinesRead += 1
			else:
				nLinesSkipped  += 1
	
	except IOError:
		am_io.log( "Error: can\'t find file " + sFile + " or read data" )
	else:
		am_io.log( "Lines: read=" + str(nLinesRead) + ", skipped=" + str(nLinesSkipped) )

	if bOpened:
		fFile.close()

	return nLinesRead
# end of def	


def filterDecisions( aJob, aCandles, aDecisions, iFile ):

	nDecisions = len( aDecisions )
	for iDecision in range( nDecisions ):
	
		if 'bDecisionToClose' in aDecisions[iDecision]:
			if aDecisions[iDecision]['bDecisionToClose'] == True:
				continue
		
		iSideToOpen = aDecisions[iDecision]['iSide']
		# Filtering side of the trade
		if aJob['iSideOfTrades'] == 1 and iSideToOpen != 1:			
			continue 
		if aJob['iSideOfTrades'] == -1 and iSideToOpen != 2:
			continue 

		iPositionId = aDecisions[iDecision]['iPositionId']
		# Searching for the closing of the position with id='iPositionId'
		iDecisionToClose = -1
		for iDecision2 in range( iDecision+1, nDecisions):
			if aDecisions[iDecision2]['iPositionId'] == iPositionId:
				aDecisions[iDecision2]['bDecisionToClose'] = True
				iDecisionToClose = iDecision2
				break
		if iDecisionToClose == -1: # If the trade hasn't been closed...
			# aJob['am_io'].log( "Position id = " + str(iPositionId) + " Hasn't been closed!", _bToFile = False, _bToStdout = True )
			continue
		iOpenDateTime = aDecisions[iDecision]['iDateTime']
		fOpenPrice = aDecisions[iDecision]['fPrice']
		iCloseDateTime = aDecisions[iDecisionToClose]['iDateTime']
		fClosePrice = aDecisions[iDecisionToClose]['fPrice']

		fProfit = 0
		if iSideToOpen == 1:
			fProfit = fClosePrice - fOpenPrice
		if iSideToOpen == 2:
			fProfit = fOpenPrice - fClosePrice
		if aJob['iProfitOfTrades'] == 1 and not( fProfit > 0 ):
			continue
		if aJob['iProfitOfTrades'] == -1 and not( fProfit < 0 ):
			continue
		aDecisions[iDecision]['fProfit'] = fProfit
		aDecisions[iDecision]['fClosePrice'] = fClosePrice
		aDecisions[iDecision]['iCloseDateTime'] = iCloseDateTime

		# To find candles to open and close the trade
		nCandles = len(aCandles)
		iCandleSt = -1
		iCandleEn = -1
		for iCandle in range( nCandles ):
			if aCandles[iCandle]['iDateTime'] >= iOpenDateTime:
				iCandleSt = iCandle
				break
		if iCandleSt == -1:
			continue
		for iCandle in range( iCandleSt, nCandles ):
			if aCandles[iCandle]['iDateTime'] >= iCloseDateTime:
				iCandleEn = iCandle
				break
		if iCandleEn == -1:
			continue

		# aJob['am_io'].log( "Warning: no candles for a decision", _bToFile = False, _bToStdout = True )  
		aDecisions[iDecision]['iCandleSt'] = iCandleSt
		aDecisions[iDecision]['iCandleEn'] = iCandleEn

		aDecisions[iDecision]['iAlgorithm'] = iFile

		aDecisions[iDecision]['bFilterOk'] = True
	# end of for

	# Removing unnecessary decisions
	for iDecision in range( nDecisions-1, -1, -1 ):
		if not( 'bFilterOk' in aDecisions[iDecision] ):
			del aDecisions[iDecision]
# end of def	


def strToDateTime( sDateTime ):
	
	iDateTime = None
	dtDateTime = None

	reDateTime = re.match( r"([0-9][0-9][0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9][0-9])", sDateTime, re.M|re.I )
	if reDateTime:
		sYear = reDateTime.group(1)
		sMonth = reDateTime.group(2)
		sDay = reDateTime.group(3)
		sH = reDateTime.group(4)
		sM = reDateTime.group(5)
		sS = reDateTime.group(6)
		iYear = int ( sYear )
		iMonth = int ( sMonth )
		iDay = int ( sDay )
		iH = int (sH )
		iM = int (sM)
		iS = int (sS)
		dtDateTime = datetime.datetime( iYear, iMonth, iDay, iH, iM, iS )
		iDateTime = dtToInt( dtDateTime )

	return (iDateTime, dtDateTime)
# end of def

'''
def loadCashedDecisions( aJob, aDecisions )
	sCachFile = aJob['sReportDir'] + aJob['os'].sep + aJob['sDataSourceFile'] + ".svd"
'''

def readCache( aJob ):
	dPP = None
	sDataSourceFile = aJob['sDataSourceFile']
	sDataCacheFile = getCacheFileName( aJob )
	if aJob['os'].path.isfile( sDataCacheFile ):
		iDataSourceMTime = aJob['os'].path.getmtime(sDataSourceFile)
		iDataCacheMTime = aJob['os'].path.getmtime(sDataCacheFile)
		if iDataSourceMTime < iDataCacheMTime:
			aJob['am_io'].log(  "Cache file (" + sDataCacheFile + ") found. Reading cached data..." )			
			import pickle
			try:
				dPP = pickle.load( open(sDataCacheFile,"rb") )
			except IOError:
				aJob['am_io'].log(  "Can't read cached data. Recalculating..." )
				dPP = None

	return dPP
# end of def

def writeCache( aJob, dPP ):
	sDataCacheFile = getCacheFileName( aJob )
	aJob['am_io'].log(  "Writing cache for future use into " + sDataCacheFile )			
	import pickle
	try:
		pickle.dump( dPP, open( sDataCacheFile, "wb" ) )
	except IOError:
		aJob['am_io'].log( "Can't write cache!" )			
# end of def

def getCacheFileName( aJob ):
	return aJob['os'].path.dirname( aJob['aFilesWithDecisions'][0] ) + aJob['os'].sep + aJob['os'].path.basename(aJob['sDataSourceFile']) + ".cch"
# end of def
