Option Explicit

' DeepSeek API 配置
Private Const API_KEY As String = "你的DeepSeek_API密钥" ' 请替换为您的实际API密钥
Private Const API_URL As String = "https://api.deepseek.com/v1/chat/completions"

' 主要函数：与DeepSeek对话并处理Excel数据
Sub ChatWithDeepSeek()
    Dim userInput As String
    Dim response As String
    Dim ws As Worksheet
    
    ' 设置工作表和输入区域
    Set ws = ThisWorkbook.Sheets("对话记录")
    If ws Is Nothing Then
        Set ws = CreateDialogSheet
    End If
    
    ' 获取用户输入
    userInput = InputBox("请输入您的问题或指令：" & vbCrLf & _
                        "例如：" & vbCrLf & _
                        "- 分析A列数据的统计信息" & vbCrLf & _
                        "- 对B列数据进行分类汇总" & vbCrLf & _
                        "- 生成数据报告", "DeepSeek Excel助手")
    
    If userInput = "" Then Exit Sub
    
    ' 添加数据上下文到请求中
    userInput = AddDataContext(userInput)
    
    ' 调用DeepSeek API
    response = CallDeepSeekAPI(userInput)
    
    ' 处理并显示结果
    If response <> "" Then
        ProcessAPIResponse response, ws
        MsgBox "操作完成！结果已记录在" & ws.Name & "工作表中。", vbInformation
    Else
        MsgBox "API调用失败，请检查网络连接和API密钥。", vbExclamation
    End If
End Sub

' 创建对话记录工作表
Function CreateDialogSheet() As Worksheet
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets.Add
    ws.Name = "对话记录"
    
    ' 设置表头
    With ws
        .Range("A1").Value = "时间"
        .Range("B1").Value = "用户输入"
        .Range("C1").Value = "AI响应"
        .Range("D1").Value = "执行结果"
        .Range("A1:D1").Font.Bold = True
        .Range("A1:D1").Interior.Color = RGB(200, 230, 255)
    End With
    
    Set CreateDialogSheet = ws
End Function

' 添加数据上下文到用户输入
Function AddDataContext(userInput As String) As String
    Dim dataContext As String
    Dim ws As Worksheet
    
    On Error Resume Next
    Set ws = ActiveSheet
    If ws Is Nothing Then
        AddDataContext = userInput
        Exit Function
    End If
    
    ' 获取当前工作表的基本信息
    With ws
        dataContext = "当前工作表数据概况：" & vbCrLf
        dataContext = dataContext & "工作表名称：" & .Name & vbCrLf
        dataContext = dataContext & "使用范围：" & .UsedRange.Address & vbCrLf
        dataContext = dataContext & "数据行数：" & .UsedRange.Rows.Count & vbCrLf
        dataContext = dataContext & "数据列数：" & .UsedRange.Columns.Count & vbCrLf
        
        ' 添加列标题信息
        If .UsedRange.Rows.Count > 1 Then
            dataContext = dataContext & "列标题：" & vbCrLf
            Dim i As Integer
            For i = 1 To .UsedRange.Columns.Count
                If .Cells(1, i).Value <> "" Then
                    dataContext = dataContext & "列" & i & ": " & .Cells(1, i).Value & vbCrLf
                End If
            Next i
        End If
    End With
    
    AddDataContext = dataContext & vbCrLf & "用户问题：" & userInput
End Function

' 调用DeepSeek API
Function CallDeepSeekAPI(prompt As String) As String
    Dim http As Object
    Dim jsonBody As String
    Dim response As String
    
    On Error GoTo ErrorHandler
    
    ' 创建HTTP请求对象
    Set http = CreateObject("MSXML2.XMLHTTP")
    
    ' 构建JSON请求体
    jsonBody = BuildJSONRequest(prompt)
    
    ' 发送POST请求
    With http
        .Open "POST", API_URL, False
        .setRequestHeader "Content-Type", "application/json"
        .setRequestHeader "Authorization", "Bearer " & API_KEY
        .send jsonBody
        
        If .Status = 200 Then
            response = .responseText
            CallDeepSeekAPI = ParseAPIResponse(response)
        Else
            CallDeepSeekAPI = ""
        End If
    End With
    
    Exit Function
    
ErrorHandler:
    CallDeepSeekAPI = ""
    MsgBox "API调用错误: " & Err.Description, vbExclamation
End Function

' 构建JSON请求
Function BuildJSONRequest(prompt As String) As String
    Dim json As String
    
    json = "{"
    json = json & """model"": ""deepseek-chat"","
    json = json & """messages"": ["
    json = json & "{""role"": ""system"", ""content"": ""你是一个Excel专家，能够分析和处理电子表格数据。请根据用户提供的数据上下文，给出具体的数据处理建议、分析结果或操作步骤。""},"
    json = json & "{""role"": ""user"", ""content"": """ & EscapeJSON(prompt) & """}"
    json = json & "],"
    json = json & """max_tokens"": 2000,"
    json = json & """temperature"": 0.7"
    json = json & "}"
    
    BuildJSONRequest = json
End Function

' 解析API响应
Function ParseAPIResponse(responseText As String) As String
    Dim json As Object
    On Error Resume Next
    
    ' 简单的JSON解析（如果需要更复杂的解析，可以引用JSON库）
    If InStr(responseText, """content"":""") > 0 Then
        Dim startPos As Integer
        Dim endPos As Integer
        
        startPos = InStr(responseText, """content"":""") + 11
        endPos = InStr(startPos, responseText, """")
        
        If endPos > startPos Then
            ParseAPIResponse = Mid(responseText, startPos, endPos - startPos)
            ' 处理转义字符
            ParseAPIResponse = Replace(ParseAPIResponse, "\""", """")
            ParseAPIResponse = Replace(ParseAPIResponse, "\n", vbCrLf)
            ParseAPIResponse = Replace(ParseAPIResponse, "\t", vbTab)
        Else
            ParseAPIResponse = "无法解析API响应"
        End If
    Else
        ParseAPIResponse = "API响应格式错误"
    End If
End Function

' 处理API响应并执行相应操作
Sub ProcessAPIResponse(response As String, ws As Worksheet)
    Dim lastRow As Long
    Dim analysisResult As String
    
    ' 找到最后一行
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row + 1
    
    ' 记录对话
    With ws
        .Cells(lastRow, 1).Value = Now
        .Cells(lastRow, 2).Value = GetLastUserInput()
        .Cells(lastRow, 3).Value = response
        .Cells(lastRow, 3).WrapText = True
        
        ' 根据响应内容执行相应的Excel操作
        analysisResult = ExecuteDataOperations(response)
        .Cells(lastRow, 4).Value = analysisResult
        .Cells(lastRow, 4).WrapText = True
    End With
    
    ' 自动调整列宽
    ws.Columns("A:D").AutoFit
End Function

' 执行数据操作
Function ExecuteDataOperations(instruction As String) As String
    Dim result As String
    result = "执行完成"
    
    ' 这里可以根据AI的响应执行具体的Excel操作
    ' 例如：数据分析、图表生成、格式设置等
    
    ' 简单的指令识别和执行
    If InStr(LCase(instruction), "汇总") > 0 Or InStr(LCase(instruction), "统计") > 0 Then
        result = GenerateSummaryReport()
    ElseIf InStr(LCase(instruction), "图表") > 0 Or InStr(LCase(instruction), "图形") > 0 Then
        result = CreateChart()
    ElseIf InStr(LCase(instruction), "排序") > 0 Then
        result = SortData()
    ElseIf InStr(LCase(instruction), "筛选") > 0 Then
        result = FilterData()
    End If
    
    ExecuteDataOperations = result
End Function

' 生成汇总报告
Function GenerateSummaryReport() As String
    On Error GoTo ErrorHandler
    Dim ws As Worksheet
    Set ws = ActiveSheet
    
    With ws
        ' 在数据下方添加汇总信息
        Dim lastRow As Long
        lastRow = .UsedRange.Rows.Count + 2
        
        .Cells(lastRow, 1).Value = "数据汇总报告"
        .Cells(lastRow, 1).Font.Bold = True
        .Cells(lastRow, 1).Font.Size = 14
        
        .Cells(lastRow + 1, 1).Value = "生成时间："
        .Cells(lastRow + 1, 2).Value = Now
        
        .Cells(lastRow + 2, 1).Value = "总行数："
        .Cells(lastRow + 2, 2).Value = .UsedRange.Rows.Count - 1
        
        .Cells(lastRow + 3, 1).Value = "总列数："
        .Cells(lastRow + 3, 2).Value = .UsedRange.Columns.Count
    End With
    
    GenerateSummaryReport = "汇总报告已生成"
    Exit Function
    
ErrorHandler:
    GenerateSummaryReport = "生成汇总报告时出错：" & Err.Description
End Function

' 创建图表
Function CreateChart() As String
    On Error GoErrorHandler
    Dim ws As Worksheet
    Set ws = ActiveSheet
    
    ' 简单的图表创建逻辑
    If ws.UsedRange.Columns.Count >= 2 Then
        Dim chartObj As ChartObject
        Set chartObj = ws.ChartObjects.Add(Left:=100, Width:=300, Top:=100, Height:=200)
        
        With chartObj.Chart
            .SetSourceData Source:=ws.UsedRange
            .ChartType = xlColumnClustered
            .HasTitle = True
            .ChartTitle.Text = "数据图表"
        End With
    End If
    
    CreateChart = "图表已创建"
    Exit Function
    
ErrorHandler:
    CreateChart = "创建图表时出错：" & Err.Description
End Function

' 其他辅助函数
Function EscapeJSON(text As String) As String
    EscapeJSON = Replace(text, """", "\""")
    EscapeJSON = Replace(EscapeJSON, vbCrLf, "\n")
    EscapeJSON = Replace(EscapeJSON, vbTab, "\t")
End Function

Function GetLastUserInput() As String
    ' 这里可以记录最后一次用户输入
    GetLastUserInput = "用户指令"
End Function

Function SortData() As String
    On Error GoTo ErrorHandler
    Dim ws As Worksheet
    Set ws = ActiveSheet
    
    With ws.Sort
        .SortFields.Clear
        .SortFields.Add Key:=ws.Range("A1"), Order:=xlAscending
        .SetRange ws.UsedRange
        .Header = xlYes
        .Apply
    End With
    
    SortData = "数据已排序"
    Exit Function
    
ErrorHandler:
    SortData = "排序时出错：" & Err.Description
End Function

Function FilterData() As String
    On Error GoTo ErrorHandler
    Dim ws As Worksheet
    Set ws = ActiveSheet
    
    If ws.AutoFilterMode Then
        ws.AutoFilterMode = False
    End If
    
    ws.UsedRange.AutoFilter
    
    FilterData = "筛选器已应用"
    Exit Function
    
ErrorHandler:
    FilterData = "应用筛选时出错：" & Err.Description
End Function

' 创建工具栏按钮（可选）
Sub AddDeepSeekButton()
    Dim toolbar As CommandBar
    Dim button As CommandBarButton
    
    On Error Resume Next
    Application.CommandBars("Standard").Controls("DeepSeek").Delete
    On Error GoTo 0
    
    Set button = Application.CommandBars("Standard").Controls.Add( _
        Type:=msoControlButton, Temporary:=True)
    
    With button
        .Caption = "DeepSeek助手"
        .OnAction = "ChatWithDeepSeek"
        .TooltipText = "调用DeepSeek AI处理Excel数据"
        .Style = msoButtonIconAndCaption
    End With
End Sub