//
//  BBMetalCamera.swift
//  BBMetalImage
//
//  Created by Kaibo Lu on 4/8/19.
//  Copyright © 2019 Kaibo Lu. All rights reserved.
//

import AVFoundation

/// Camera photo delegate defines handling taking photo result behaviors
public protocol BBMetalCameraPhotoDelegate: AnyObject {
  /// Called when camera did take a photo and get Metal texture
  ///
  /// - Parameters:
  ///   - camera: camera to use
  ///   - texture: Metal texture of the original photo which is not filtered
  func camera(_ camera: BBMetalCamera, didOutput texture: MTLTexture)

  /// Called when camera fail taking a photo
  ///
  /// - Parameters:
  ///   - camera: camera to use
  ///   - error: error for taking the photo
  func camera(_ camera: BBMetalCamera, didFail error: Error)
}

public protocol BBMetalCameraMetadataObjectDelegate: AnyObject {
  /// Called when camera did get metadata objects
  ///
  /// - Parameters:
  ///   - camera: camera to use
  ///   - metadataObjects: metadata objects
  func camera(_ camera: BBMetalCamera, didOutput metadataObjects: [AVMetadataObject])
}

public enum BBMetalCameraError: Error {
  case cannotAddAudioInput
  case cannotAddAudioOutput
  case noVideoDevice
  case cannotCreateVideoInput
  case cannotAddVideoInput
  case cannotAddVideoOutput
  case videoOrientationNotSupported
  case cannotCreateMetalTextureCache
}

// TODO: Remove when device lookup issue fix is verified
@_spi(Internals)
public enum BBMetalCameraInternalError: Error {
  case noVideoDeviceOldImpl
  case noVideoDeviceNewImpl
	case nonErrorCaptureDevicesLog([String])
}

/// Camera capturing image and providing Metal texture
public class BBMetalCamera: NSObject {
  // TODO: Remove when device lookup issue fix is verified
  @_spi(Internals)
  public func setInternalErrorHandler(_ handler: ((BBMetalCameraInternalError) -> Void)?) {
    deviceLookup.internalErrorHandler = handler
  }

  /// Image consumers
  public var consumers: [BBMetalImageConsumer] {
    lock.wait()
    let c = _consumers
    lock.signal()
    return c
  }
  private var _consumers: [BBMetalImageConsumer]

  /// A block to call before processing each video sample buffer
  public var preprocessVideo: ((CMSampleBuffer) -> Void)? {
    get {
      lock.wait()
      let p = _preprocessVideo
      lock.signal()
      return p
    }
    set {
      lock.wait()
      _preprocessVideo = newValue
      lock.signal()
    }
  }
  private var _preprocessVideo: ((CMSampleBuffer) -> Void)?

  /// A block to call before transmiting texture to image consumers
  public var willTransmitTexture: ((MTLTexture, CMTime) -> Void)? {
    get {
      lock.wait()
      let w = _willTransmitTexture
      lock.signal()
      return w
    }
    set {
      lock.wait()
      _willTransmitTexture = newValue
      lock.signal()
    }
  }
  private var _willTransmitTexture: ((MTLTexture, CMTime) -> Void)?

  /// Camera position
  public var position: AVCaptureDevice.Position { return camera.position }

  /// Camera active format
  public var activeFormat: AVCaptureDevice.Format { camera.activeFormat }

  /// Output texture size
  public var textureSize: BBMetalIntSize {
    lock.wait()
    let dimensions = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
    var size = BBMetalIntSize(width: Int(dimensions.width), height: Int(dimensions.height))
    if let orientation = videoOutput.connections.first?.videoOrientation {
      let isLandscape: (AVCaptureVideoOrientation) -> Bool = { orientation in
        orientation == .landscapeLeft || orientation == .landscapeRight
      }
      if isLandscape(orientation) != isLandscape(originalOrientation) {
        swap(&size.width, &size.height)
      }
    }
    lock.signal()
    return size
  }

  public var videoOrientation: AVCaptureVideoOrientation {
    get {
      lock.wait()
      let v = videoOutput.connections.first?.videoOrientation ?? .portrait
      lock.signal()
      return v
    }
    set {
      lock.wait()
      if let connection = videoOutput.connections.first,
        connection.isVideoOrientationSupported
      {
        connection.videoOrientation = newValue
      }
      lock.signal()
    }
  }

  private var originalOrientation: AVCaptureVideoOrientation

  /// Whether to run benchmark or not.
  /// Running benchmark records frame duration.
  /// False by default.
  public var benchmark: Bool {
    get {
      lock.wait()
      let b = _benchmark
      lock.signal()
      return b
    }
    set {
      lock.wait()
      _benchmark = newValue
      lock.signal()
    }
  }
  private var _benchmark: Bool

  /// Average frame duration, or 0 if not valid value.
  /// To get valid value, set `benchmark` to true.
  public var averageFrameDuration: Double {
    lock.wait()
    let d =
      capturedFrameCount > ignoreInitialFrameCount
      ? totalCaptureFrameTime / Double(capturedFrameCount - ignoreInitialFrameCount) : 0
    lock.signal()
    return d
  }

  /// true means it's separate audio session
  public var sessionInterruptionHandler:
    ((AVCaptureSession, AVCaptureSession.InterruptionReason, Bool) -> Void)?

  /// Called when an interruption ends and the session is resumed.
  /// true means it's separate audio session
  public var sessionInterruptionEndedHandler: ((AVCaptureSession, Bool) -> Void)?

  /// true means it's separate audio session
  public var sessionErrorHandler: ((AVCaptureSession, AVError, Bool) -> Void)?

  /// Called when the camera resumes a session on its own after an interruption or runtime error.
  public var sessionDidRecoverHandler: ((AVCaptureSession) -> Void)?

  private static let maxSessionRestartAttempts = 3
  /// The system resumes most interrupted sessions itself. Waiting before stepping in keeps the two
  /// from restarting the same session at once, which is visible as a flickering preview.
  private static let sessionResumeGracePeriod: TimeInterval = 0.5
  /// Hard floor between two resume attempts, so no interruption pattern can turn into a restart loop.
  private static let minimumSessionResumeInterval: CFTimeInterval = 2.0
  /// Zoom ramps change the factor by 2^rate per second, so one doubling takes ~0.17s here.
  private static let zoomRampRate: Float = 6.0

  /// Shared on purpose: in multi-cam both cameras observe the same session, so restarts have to be
  /// serialized across instances rather than racing two `startRunning` calls on one session.
  private static let sessionRestartQueue = DispatchQueue(
    label: "com.Kaibo.BBMetalImage.Camera.sessionRestart", qos: .userInitiated)
  /// Whether the app still wants this camera running. Recovery only resumes sessions the app
  /// has not explicitly stopped.
  private var isRunningRequested: Bool
  private var sessionRestartAttempt: Int
  private var lastSessionResumeTime: CFTimeInterval
  private var lastInterruptionReason: AVCaptureSession.InterruptionReason?
  private var observedSessions: Set<ObjectIdentifier>

  private var capturedFrameCount: Int
  private var totalCaptureFrameTime: Double
  private let ignoreInitialFrameCount: Int

  private let lock: DispatchSemaphore

  private let deviceLookup: DeviceLookup = .init()
  private var session: AVCaptureSession!
  private var camera: AVCaptureDevice!
  private var videoInput: AVCaptureDeviceInput!
  private var videoOutput: AVCaptureVideoDataOutput!
  private var videoOutputQueue: DispatchQueue!

  private let multitpleSessions: Bool
  private var audioSession: AVCaptureSession?
  private var audioInput: AVCaptureDeviceInput?
  private var audioOutput: AVCaptureAudioDataOutput?
  private var audioOutputQueue: DispatchQueue!
  // Guards the audio graph so build/start/teardown run OUTSIDE `lock`: the video captureOutput
  // takes `lock` every frame, and a cold attach under it froze the preview for ~0.5s.
  // Ordering rule: `lock` may nest `audioGraphLock`, never the reverse.
  private let audioGraphLock = DispatchSemaphore(value: 1)

  /// Audio consumer processing audio sample buffer.
  /// Set this property to nil (default value) if not recording audio.
  /// Set this property to a given audio consumer if recording audio.
  public var audioConsumer: BBMetalAudioConsumer? {
    get {
      lock.wait()
      let a = _audioConsumer
      lock.signal()
      return a
    }
    set {
      // Audio graph work runs outside `lock` (see `audioGraphLock`); only the consumer swap is
      // under it.
      if newValue != nil {
        let error = ensureAudioCaptureRunning()
        lock.wait()
        audioConsumerAssignError = error
        _audioConsumer = error == nil ? newValue : nil
        lock.signal()
      } else {
        lock.wait()
        audioConsumerAssignError = nil
        _audioConsumer = nil
        lock.signal()
        removeAudioInputAndOutput()
      }
    }
  }
  public var audioConsumerAssignError: Error?
  private var _audioConsumer: BBMetalAudioConsumer?

  private var photoOutput: AVCapturePhotoOutput!

  /// Whether can take photo or not.
  /// Set this property to true before calling `takePhoto()` method.
  public var canTakePhoto: Bool {
    get {
      lock.wait()
      let c = _canTakePhoto
      lock.signal()
      return c
    }
    set {
      lock.wait()
      _canTakePhoto = newValue
      if newValue {
        if !addPhotoOutput() { _canTakePhoto = false }
      } else {
        removePhotoOutput()
      }
      lock.signal()
    }
  }
  private var _canTakePhoto: Bool

  /// Opt-in: prefer an active format that can deliver large stills, and request them from the photo
  /// output. Off by default because a still larger than the active video format makes the sensor
  /// switch readout modes on every capture, which costs hundreds of milliseconds of shutter latency.
  /// Set this before `setFrameRate(_:)` so format selection can take it into account.
  public var prefersHighResolutionStills: Bool = false

  /// Camera photo delegate handling taking photo result.
  /// To take photo, this property should not be nil.
  public weak var photoDelegate: BBMetalCameraPhotoDelegate? {
    get {
      lock.wait()
      let p = _photoDelegate
      lock.signal()
      return p
    }
    set {
      lock.wait()
      _photoDelegate = newValue
      lock.signal()
    }
  }
  private weak var _photoDelegate: BBMetalCameraPhotoDelegate?

  private var _needPhoto: Bool

  private var _capturePhotoCompletion: BBMetalFilterCompletion?

  private var metadataOutput: AVCaptureMetadataOutput!
  private var metadataOutputQueue: DispatchQueue!

  public weak var metadataObjectDelegate: BBMetalCameraMetadataObjectDelegate? {
    get {
      lock.wait()
      let m = _metadataObjectDelegate
      lock.signal()
      return m
    }
    set {
      lock.wait()
      _metadataObjectDelegate = newValue
      lock.signal()
    }
  }
  private weak var _metadataObjectDelegate: BBMetalCameraMetadataObjectDelegate?

  /// When this property is false, received video/audio sample buffer will not be processed
  public var isPaused: Bool {
    get {
      lock.wait()
      let p = _isPaused
      lock.signal()
      return p
    }
    set {
      lock.wait()
      _isPaused = newValue
      lock.signal()
    }
  }
  private var _isPaused: Bool

  private var textureCache: CVMetalTextureCache!

  public var currentDeviceDisplayVideoZoomFactorMultiplier: CGFloat {
    if #available(iOS 18.0, *) {
      return videoInput?.device.displayVideoZoomFactorMultiplier ?? 1.0
    } else {
      return 1.0
    }
  }

  public var isUltraWideBackCameraSupported: Bool {
    AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) != nil
  }

  public var currentAbsoluteZoomFactor: CGFloat {
    camera.videoZoomFactor
  }

  /// Creates a camera
  /// - Parameters:
  ///   - sessionPreset: a constant value indicating the quality level or bit rate of the output
  ///   - position: camera position
  ///   - multitpleSessions: whether to use independent video session and audio session (false by default). Switching camera position while recording leads to the video and audio out of sync.
  /// Set true if we allow the user to switch camera position while recording.
  public convenience init(
    captureSession: AVCaptureSession = .init(),
    sessionPreset: AVCaptureSession.Preset = .high,
    position: AVCaptureDevice.Position = .back,
    multitpleSessions: Bool = false,
  ) throws {
    try self.init(
      captureSession: captureSession,
      sessionPreset: sessionPreset,
      position: position,
      multitpleSessions: multitpleSessions,
      internalErrorHandler: nil
    )
  }

  /// Creates a camera
  /// - Parameters:
  ///   - sessionPreset: a constant value indicating the quality level or bit rate of the output
  ///   - position: camera position
  ///   - multitpleSessions: whether to use independent video session and audio session (false by default). Switching camera position while recording leads to the video and audio out of sync.
  /// Set true if we allow the user to switch camera position while recording.
  @_spi(Internals)
  public init(
    captureSession: AVCaptureSession = .init(),
    sessionPreset: AVCaptureSession.Preset = .high,
    position: AVCaptureDevice.Position = .back,
    multitpleSessions: Bool = false,
    internalErrorHandler: ((BBMetalCameraInternalError) -> Void)?
  ) throws {
    _consumers = []
    _canTakePhoto = false
    _needPhoto = false
    _isPaused = false
    _benchmark = false
    capturedFrameCount = 0
    totalCaptureFrameTime = 0
    ignoreInitialFrameCount = 5
    originalOrientation = .portrait
    isRunningRequested = false
    sessionRestartAttempt = 0
    lastSessionResumeTime = 0
    observedSessions = []
    self.multitpleSessions = multitpleSessions
    lock = DispatchSemaphore(value: 1)

    super.init()
    setInternalErrorHandler(internalErrorHandler)

    let devices = deviceLookup.session.devices
    internalErrorHandler?(.nonErrorCaptureDevicesLog(devices.map { device in
      device.deviceType.rawValue
    }))

    let videoDevice: AVCaptureDevice?
    if captureSession is AVCaptureMultiCamSession {
      videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
        ?? deviceLookup.device(for: position)
    } else {
      videoDevice = deviceLookup.device(for: position)
    }
    guard let videoDevice else {
      throw BBMetalCameraError.noVideoDevice
    }

    guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else {
      throw BBMetalCameraError.cannotCreateVideoInput
    }

    session = captureSession
    session.automaticallyConfiguresApplicationAudioSession = false
    session.beginConfiguration()

    if !(session is AVCaptureMultiCamSession) {
      session.sessionPreset = sessionPreset
    }

    if !session.canAddInput(videoDeviceInput) {
      session.commitConfiguration()
      throw BBMetalCameraError.cannotAddVideoInput
    }

    session.addInput(videoDeviceInput)
    camera = videoDevice
    videoInput = videoDeviceInput

    let videoDataOutput = AVCaptureVideoDataOutput()
    videoDataOutput.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]
    videoDataOutput.alwaysDiscardsLateVideoFrames = true
    videoOutputQueue = DispatchQueue(label: "com.Kaibo.BBMetalImage.Camera.videoOutput")
    videoDataOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
    if !session.canAddOutput(videoDataOutput) {
      session.commitConfiguration()
      throw BBMetalCameraError.cannotAddVideoOutput
    }
    session.addOutput(videoDataOutput)
    videoOutput = videoDataOutput

    guard let connection = videoDataOutput.connections.first,
      connection.isVideoOrientationSupported
    else {
      session.commitConfiguration()
      throw BBMetalCameraError.videoOrientationNotSupported
    }
    originalOrientation = connection.videoOrientation
    connection.videoOrientation = .portrait
    configureVideoStabilization(for: connection)

    session.commitConfiguration()

    #if !targetEnvironment(simulator)
      if CVMetalTextureCacheCreate(
        kCFAllocatorDefault, nil, BBMetalDevice.sharedDevice, nil, &textureCache)
        != kCVReturnSuccess || textureCache == nil
      {
        throw BBMetalCameraError.cannotCreateMetalTextureCache
      }
    #endif
  }

  /// Builds the audio capture graph without starting it. Route-neutral: the microphone does not
  /// appear in the audio route until the session runs, so other apps' playback does not dip.
  public func prewarmAudioCapture() {
    audioGraphLock.wait()
    defer { audioGraphLock.signal() }
    if let error = buildAudioGraph() {
      print("Audio capture prewarm failed: \(error)")
    }
  }

  private func ensureAudioCaptureRunning() -> Error? {
    audioGraphLock.wait()
    defer { audioGraphLock.signal() }
    if let error = buildAudioGraph() { return error }
    if multitpleSessions, let audioSession, !audioSession.isRunning, session.isRunning {
      audioSession.startRunning()
    }
    return nil
  }

  /// Must be called with `audioGraphLock` held.
  private func buildAudioGraph() -> Error? {
    if audioOutput != nil { return nil }

    var session: AVCaptureSession = self.session
    if multitpleSessions {
      session = AVCaptureSession()
      session.automaticallyConfiguresApplicationAudioSession = false
      audioSession = session
    }

    session.beginConfiguration()
    defer { session.commitConfiguration() }

    guard let audioDevice = AVCaptureDevice.default(for: .audio),
      let input = try? AVCaptureDeviceInput(device: audioDevice),
      session.canAddInput(input)
    else {
      print("Can not add audio input")
      return BBMetalCameraError.cannotAddAudioInput
    }
    session.addInput(input)
    audioInput = input

    let output = AVCaptureAudioDataOutput()
    let outputQueue = DispatchQueue(label: "com.Kaibo.BBMetalImage.Camera.audioOutput")
    output.setSampleBufferDelegate(self, queue: outputQueue)
    guard session.canAddOutput(output) else {
      _removeAudioInputAndOutput()
      print("Can not add audio output")
      return BBMetalCameraError.cannotAddAudioOutput
    }
    session.addOutput(output)
    audioOutput = output
    audioOutputQueue = outputQueue

    return nil
  }

  private func removeAudioInputAndOutput() {
    audioGraphLock.wait()
    defer { audioGraphLock.signal() }
    // With nothing to tear down this used to reconfigure the video session anyway. On a live
    // multi-cam session that alone re-provisions hardware and makes both previews re-run their
    // exposure ramp, and the dual-camera path clears the secondary camera's audio on every attach.
    guard audioInput != nil || audioOutput != nil else { return }
    // Audio lives in its own session when `multitpleSessions` is set; reconfiguring the video
    // session in that case would disturb capture for no reason.
    let configuredSession: AVCaptureSession? = multitpleSessions ? audioSession : session
    configuredSession?.beginConfiguration()
    _removeAudioInputAndOutput()
    configuredSession?.commitConfiguration()
  }

  private func _removeAudioInputAndOutput() {
    let session: AVCaptureSession? = multitpleSessions ? audioSession : self.session
    if let input = audioInput {
      session?.removeInput(input)
      audioInput = nil
    }
    if let output = audioOutput {
      session?.removeOutput(output)
      audioOutput = nil
    }
    if audioOutputQueue != nil {
      audioOutputQueue = nil
    }
    if audioSession != nil {
      audioSession = nil
    }
  }

  @discardableResult
  private func addPhotoOutput() -> Bool {
    if photoOutput != nil { return true }

    session.beginConfiguration()
    defer { session.commitConfiguration() }

    let output = AVCapturePhotoOutput()
    if !session.canAddOutput(output) {
      print("Can not add photo output")
      return false
    }
    session.addOutput(output)
    photoOutput = output

    output.maxPhotoQualityPrioritization = .balanced
    if let dimensions = preferredPhotoDimensions() {
      output.maxPhotoDimensions = dimensions
    }
    if #available(iOS 17.0, *) {
      // Zero shutter lag captures the frame from the moment the shutter was pressed rather than the
      // one that lands after the request travels through the pipeline.
      if output.isZeroShutterLagSupported { output.isZeroShutterLagEnabled = true }
      if output.isResponsiveCaptureSupported { output.isResponsiveCaptureEnabled = true }
    }

    return true
  }

  /// Still dimensions to request. The app publishes at most 1920px on the long side, so the smallest
  /// supported size that comfortably clears that is preferred over the sensor maximum: it keeps the
  /// one-shot buffer (and the filter chain's intermediate textures) several times smaller for the
  /// same delivered quality. Falls back to the largest available when there is no such option.
  private func preferredPhotoDimensions() -> CMVideoDimensions? {
    guard prefersHighResolutionStills else { return nil }

    let supported = camera.activeFormat.supportedMaxPhotoDimensions
    guard !supported.isEmpty else { return nil }

    let minimumLongSide: Int32 = 2560
    let ascending = supported.sorted {
      Int($0.width) * Int($0.height) < Int($1.width) * Int($1.height)
    }
    return ascending.first { max($0.width, $0.height) >= minimumLongSide } ?? ascending.last
  }

  private static func maxStillPixelCount(of format: AVCaptureDevice.Format) -> Int {
    format.supportedMaxPhotoDimensions.reduce(0) { partial, dimensions in
      max(partial, Int(dimensions.width) * Int(dimensions.height))
    }
  }

  /// Supported still sizes belong to the active format, so this has to be re-applied whenever the
  /// format changes — on frame-rate selection and after switching to the other camera.
  private func updatePhotoOutputDimensions() {
    guard let output = photoOutput, let dimensions = preferredPhotoDimensions() else { return }
    output.maxPhotoDimensions = dimensions
  }

  private func removePhotoOutput() {
    session.beginConfiguration()
    if let output = photoOutput { session.removeOutput(output) }
    session.commitConfiguration()
  }

  @discardableResult
  public func addMetadataOutput(with types: [AVMetadataObject.ObjectType]) -> Bool {
    var result = false

    lock.wait()

    if metadataOutput != nil {
      lock.signal()
      return result
    }

    session.beginConfiguration()

    let output = AVCaptureMetadataOutput()
    let outputQueue = DispatchQueue(label: "com.Kaibo.BBMetalImage.Camera.metadataOutput")
    output.setMetadataObjectsDelegate(self, queue: outputQueue)

    if session.canAddOutput(output) {
      session.addOutput(output)
      let validTypes = types.filter { output.availableMetadataObjectTypes.contains($0) }
      output.metadataObjectTypes = validTypes
      metadataOutput = output
      metadataOutputQueue = outputQueue
      result = true
    }

    session.commitConfiguration()
    lock.signal()
    return result
  }

  public func removeMetadataOutput() {
    lock.wait()

    if metadataOutput == nil {
      lock.signal()
      return
    }

    session.beginConfiguration()

    session.removeOutput(metadataOutput)
    metadataOutput = nil
    metadataOutputQueue = nil

    session.commitConfiguration()
    lock.signal()
  }

  /// Captures frame texture as a photo.
  /// Get original frame texture in the completion closure.
  /// To get filtered texture, use `addCompletedHandler(_:) ` method of `BBMetalBaseFilter`, check whether the filtered texture is camera photo.
  /// This method is much faster than `takePhoto()` method.
  /// - Parameter completion: a closure to call after capturing. If success, get original frame texture. If failure, get error.
  public func capturePhoto(completion: BBMetalFilterCompletion? = nil) {
    lock.wait()
    _needPhoto = true
    _capturePhotoCompletion = completion
    lock.signal()
  }

  /// Takes a photo.
  /// Before calling this method, set `canTakePhoto` property to true and `photoDelegate` property to nonnull.
  /// Get original frame texture in `camera(_:didOutput:)` method of `BBMetalCameraPhotoDelegate`.
  /// To get filtered texture, use `capturePhoto(completion:)` method, or create new filter to process the original frame texture.
  public func takePhoto(
    flashMode: AVCaptureDevice.FlashMode = .off
  ) {
    lock.wait()
    if let output = photoOutput, _photoDelegate != nil {
      // Uncompressed BGRA keeps the still in a form the Metal filter chain can consume directly.
      let currentSettings = AVCapturePhotoSettings(format: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ])
      // `.speed` on purpose: multi-frame processing (Deep Fusion, Smart HDR) does not run for
      // uncompressed captures, so anything above `.speed` costs shutter latency and returns nothing.
      currentSettings.photoQualityPrioritization = .speed
      currentSettings.maxPhotoDimensions = output.maxPhotoDimensions
      if camera.hasFlash {
        currentSettings.flashMode = flashMode
      }
      output.capturePhoto(with: currentSettings, delegate: self)
    }
    lock.signal()
  }

  /// Switches camera position (back to front, or front to back)
  ///
  /// - Returns: true if succeed, or false if fail
  @discardableResult
  public func switchCameraPosition() -> Bool {
    lock.wait()
    session.beginConfiguration()
    defer {
      session.commitConfiguration()
      lock.signal()
    }

    var position: AVCaptureDevice.Position = .back
    if camera.position == .back { position = .front }

    guard
      let videoDevice = deviceLookup.device(for: position),
      let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice)
    else { return false }

    session.removeInput(videoInput)

    guard session.canAddInput(videoDeviceInput) else {
      session.addInput(videoInput)
      return false
    }
    session.addInput(videoDeviceInput)
    camera = videoDevice
    videoInput = videoDeviceInput

    guard let connection = videoOutput.connections.first,
      connection.isVideoOrientationSupported
    else { return false }
    originalOrientation = connection.videoOrientation
    connection.videoOrientation = .portrait
    configureVideoStabilization(for: connection)
    updatePhotoOutputDimensions()

    return true
  }

  /// Stabilization is applied to the video data output, so it benefits both the live preview and
  /// anything recorded from it. `.standard` is deliberate: the cinematic modes buffer several
  /// frames ahead, which reads as preview lag in a realtime filter pipeline. Multi-cam sessions are
  /// skipped because stabilization raises `hardwareCost` and can push the session over budget.
  private func configureVideoStabilization(for connection: AVCaptureConnection) {
    if session is AVCaptureMultiCamSession { return }
    guard camera.activeFormat.isVideoStabilizationModeSupported(.standard) else { return }
    connection.preferredVideoStabilizationMode = .standard
  }

  public func setRelativeZoomFactor(_ relativeZoom: CGFloat, animated: Bool = false) {
    let absoluteZoom = relativeZoom / currentDeviceDisplayVideoZoomFactorMultiplier

    self.setAbsoluteZoomFactor(absoluteZoom, animated: animated)
  }

  /// - Parameter animated: ramps to the target instead of jumping to it. Use it for discrete zoom
  ///   steps (0.5x/1x/2x); a continuous gesture must stay unanimated, since restarting a ramp on
  ///   every gesture update lags behind the finger.
  public func setAbsoluteZoomFactor(_ absoluteZoom: CGFloat, animated: Bool = false) {
    let clampedZoom = max(
      camera.minAvailableVideoZoomFactor, min(absoluteZoom, camera.maxAvailableVideoZoomFactor))

    self.configureCamera {
      if animated {
        $0.ramp(toVideoZoomFactor: clampedZoom, withRate: BBMetalCamera.zoomRampRate)
      } else {
        if $0.isRampingVideoZoom { $0.cancelVideoZoomRamp() }
        $0.videoZoomFactor = clampedZoom
      }
    }
  }

  /// Sets camera frame rate
  ///
  /// - Parameter frameRate: camera frame rate
  /// - Returns: true if succeed, or false if fail
  @discardableResult
  public func setFrameRate(_ frameRate: Float64) -> Bool {
    var success = false
    lock.wait()
    do {
      try camera.lockForConfiguration()
      let dimensions = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
      let candidates = camera.formats.filter { format in
        let newDimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        guard dimensions.width == newDimensions.width,
          dimensions.height == newDimensions.height
        else { return false }
        return format.videoSupportedFrameRateRanges.contains { range in
          range.maxFrameRate >= frameRate && range.minFrameRate <= frameRate
        }
      }

      // Every candidate delivers the same video dimensions, so preferring the one with the largest
      // still support raises photo resolution without changing the preview or recording pipeline.
      var targetFormat = candidates.first
      if prefersHighResolutionStills {
        targetFormat =
          candidates.max { lhs, rhs in
            BBMetalCamera.maxStillPixelCount(of: lhs) < BBMetalCamera.maxStillPixelCount(of: rhs)
          } ?? candidates.first
      }

      if let format = targetFormat {
        camera.activeFormat = format
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(frameRate))
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(frameRate))
        updatePhotoOutputDimensions()
        success = true
      } else {
        print("Can not find valid format for camera frame rate \(frameRate)")
      }
      camera.unlockForConfiguration()
    } catch {
      print("Error for camera lockForConfiguration: \(error)")
    }
    lock.signal()
    return success
  }

  /// Configures camera.
  /// Configure camera in the block, without calling `lockForConfiguration` and `unlockForConfiguration` methods.
  ///
  /// - Parameter block: closure configuring camera
  public func configureCamera(_ block: (AVCaptureDevice) -> Void) {
    lock.wait()
    do {
      try camera.lockForConfiguration()
      block(camera)
      camera.unlockForConfiguration()
    } catch {
      print("Error for camera lockForConfiguration: \(error)")
    }
    lock.signal()
  }

  /// Starts capturing
  public func start() {
    lock.wait()
    isRunningRequested = true
    sessionRestartAttempt = 0
    addObservers(session: session)
    session.startRunning()
    if multitpleSessions {
      audioGraphLock.wait()
      // A prewarmed graph (no consumer attached) must stay stopped: running audio I/O puts the
      // mic into the audio route and dips other apps' playback.
      if let session = audioSession, _audioConsumer != nil {
        addObservers(session: session)
        session.startRunning()
      }
      audioGraphLock.signal()
    }
    lock.signal()
  }

  /// Stops capturing
  public func stop() {
    lock.wait()
    isRunningRequested = false
    removeObservers(session: session)
    session.stopRunning()
    if multitpleSessions {
      audioGraphLock.wait()
      if let session = audioSession {
        removeObservers(session: session)
        session.stopRunning()
      }
      audioGraphLock.signal()
    }
    lock.signal()
  }

  /// Resets benchmark record data
  public func resetBenchmark() {
    lock.wait()
    capturedFrameCount = 0
    totalCaptureFrameTime = 0
    lock.signal()
  }
}

extension BBMetalCamera: BBMetalImageSource {
  @discardableResult
  public func add<T: BBMetalImageConsumer>(consumer: T) -> T {
    lock.wait()
    _consumers.append(consumer)
    lock.signal()
    consumer.add(source: self)
    return consumer
  }

  public func add(consumer: BBMetalImageConsumer, at index: Int) {
    lock.wait()
    _consumers.insert(consumer, at: index)
    lock.signal()
    consumer.add(source: self)
  }

  public func remove(consumer: BBMetalImageConsumer) {
    lock.wait()
    if let index = _consumers.firstIndex(where: { $0 === consumer }) {
      _consumers.remove(at: index)
      lock.signal()
      consumer.remove(source: self)
    } else {
      lock.signal()
    }
  }

  public func removeAllConsumers() {
    lock.wait()
    let consumers = _consumers
    _consumers.removeAll()
    lock.signal()
    for consumer in consumers {
      consumer.remove(source: self)
    }
  }
}

extension BBMetalCamera {
  /// Must be called with `lock` held. Registering twice for the same session would deliver every
  /// interruption/error callback once per registration.
  fileprivate func addObservers(session: AVCaptureSession) {
    guard observedSessions.insert(ObjectIdentifier(session)).inserted else { return }
    NotificationCenter.default.addObserver(
      self, selector: #selector(BBMetalCamera.sessionRuntimeErrorOccurred(notification:)),
      name: NSNotification.Name.AVCaptureSessionRuntimeError, object: session)
    NotificationCenter.default.addObserver(
      self, selector: #selector(BBMetalCamera.sessionWasInterrupted(notification:)),
      name: NSNotification.Name.AVCaptureSessionWasInterrupted, object: session)
    NotificationCenter.default.addObserver(
      self, selector: #selector(BBMetalCamera.sessionInterruptionEnded),
      name: NSNotification.Name.AVCaptureSessionInterruptionEnded, object: session)
  }

  /// Must be called with `lock` held.
  fileprivate func removeObservers(session: AVCaptureSession) {
    guard observedSessions.remove(ObjectIdentifier(session)) != nil else { return }
    NotificationCenter.default.removeObserver(
      self, name: NSNotification.Name.AVCaptureSessionRuntimeError, object: session)
    NotificationCenter.default.removeObserver(
      self, name: NSNotification.Name.AVCaptureSessionWasInterrupted, object: session)
    NotificationCenter.default.removeObserver(
      self, name: NSNotification.Name.AVCaptureSessionInterruptionEnded, object: session)
  }

  @objc fileprivate func sessionWasInterrupted(notification: Notification) {
    guard
      let userInfoValue = notification.userInfo?[AVCaptureSessionInterruptionReasonKey]
        as AnyObject?,
      let reasonIntegerValue = userInfoValue.integerValue,
      let reason = AVCaptureSession.InterruptionReason(rawValue: reasonIntegerValue),
      let session = notification.object as? AVCaptureSession
    else {
      return
    }
    lock.wait()
    lastInterruptionReason = reason
    lock.signal()
    let isSeparateAudioSession = multitpleSessions && session == audioSession
    sessionInterruptionHandler?(session, reason, isSeparateAudioSession)
  }

  @objc fileprivate func sessionInterruptionEnded(notification: Notification) {
    guard let session = notification.object as? AVCaptureSession else { return }
    let isSeparateAudioSession = multitpleSessions && session == audioSession
    sessionInterruptionEndedHandler?(session, isSeparateAudioSession)

    // Multi-cam renegotiates hardware while it spins up and is interrupted several times on the way,
    // and the system brings it back itself. Resuming here restarts a session that is already
    // recovering, which is visible as a flickering preview.
    if session is AVCaptureMultiCamSession { return }

    lock.wait()
    let reason = lastInterruptionReason
    lock.signal()
    // Under system pressure the fix is to shed load, not to restart: restarting is immediately
    // interrupted again.
    guard reason != .videoDeviceNotAvailableDueToSystemPressure else { return }

    // A session stopped by an interruption (phone call, another app taking the camera, Split View)
    // does not always come back on its own; resuming here is what keeps the preview from going
    // permanently black.
    resumeSessionIfNeeded(session)
  }

  @objc fileprivate func sessionRuntimeErrorOccurred(notification: Notification) {
    guard let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError,
      let session = notification.object as? AVCaptureSession
    else {
      return
    }
    let isSeparateAudioSession = multitpleSessions && session == audioSession
    sessionErrorHandler?(session, error, isSeparateAudioSession)

    // Media services reset tears the capture stack down under us; restarting the session is the
    // documented recovery. Other runtime errors are left to the app to decide on.
    guard error.code == .mediaServicesWereReset else { return }
    resumeSessionIfNeeded(session, isRetry: true)
  }

  /// Resumes a session the app still wants running. Retries are capped and backed off so a session
  /// that keeps failing does not spin forever.
  fileprivate func resumeSessionIfNeeded(_ session: AVCaptureSession, isRetry: Bool = false) {
    lock.wait()
    guard isRunningRequested else {
      lock.signal()
      return
    }
    var delay = BBMetalCamera.sessionResumeGracePeriod
    if isRetry {
      guard sessionRestartAttempt < BBMetalCamera.maxSessionRestartAttempts else {
        lock.signal()
        return
      }
      sessionRestartAttempt += 1
      delay = 0.5 * Double(sessionRestartAttempt)
    }
    lock.signal()

    // `startRunning` blocks, so it must not run on the notification thread.
    BBMetalCamera.sessionRestartQueue.asyncAfter(deadline: .now() + delay) { [weak self] in
      guard let self = self else { return }

      // Re-checked after the delay: by now the system has usually resumed the session itself.
      guard !session.isRunning else { return }

      let now = CACurrentMediaTime()
      self.lock.wait()
      let shouldRun = self.isRunningRequested
      let isTooSoon =
        now - self.lastSessionResumeTime < BBMetalCamera.minimumSessionResumeInterval
      if shouldRun, !isTooSoon {
        self.lastSessionResumeTime = now
      }
      self.lock.signal()

      guard shouldRun, !isTooSoon else { return }
      session.startRunning()

      guard session.isRunning else { return }
      self.lock.wait()
      self.sessionRestartAttempt = 0
      self.lock.signal()
      self.sessionDidRecoverHandler?(session)
    }
  }
}

extension BBMetalCamera: AVCaptureVideoDataOutputSampleBufferDelegate,
  AVCaptureAudioDataOutputSampleBufferDelegate
{
  public func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    // Audio
    if output is AVCaptureAudioDataOutput {
      lock.wait()
      let paused = _isPaused
      let currentAudioConsumer = _audioConsumer
      lock.signal()
      if !paused,
        let consumer = currentAudioConsumer
      {
        consumer.newAudioSampleBufferAvailable(sampleBuffer)
      }
      return
    }

    // Video
    lock.wait()
    let paused = _isPaused
    let consumers = _consumers
    let willTransmit = _willTransmitTexture
    let preprocessVideo = _preprocessVideo
    let cameraPosition = camera.position

    let isCameraPhoto = _needPhoto
    if _needPhoto { _needPhoto = false }

    let capturePhotoCompletion = _capturePhotoCompletion
    if _capturePhotoCompletion != nil { _capturePhotoCompletion = nil }

    let startTime = _benchmark ? CACurrentMediaTime() : 0
    lock.signal()

    guard !paused, !consumers.isEmpty else { return }

    preprocessVideo?(sampleBuffer)

    guard let texture = texture(with: sampleBuffer) else {
      if let completion = capturePhotoCompletion {
        let error = NSError(
          domain: "BBMetalCameraErrorDomain", code: 0,
          userInfo: [NSLocalizedDescriptionKey: "Can not get Metal texture"])
        let info = BBMetalFilterCompletionInfo(
          result: .failure(error),
          sampleTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
          cameraPosition: cameraPosition,
          isCameraPhoto: isCameraPhoto)
        completion(info)
      }
      return
    }

    let sampleTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

    if let completion = capturePhotoCompletion {
      var result: Result<MTLTexture, Error>
      let filter = BBMetalPassThroughFilter(createTexture: true)
      if let metalTexture = filter.filteredTexture(with: texture.metalTexture) {
        result = .success(metalTexture)
      } else {
        let error = NSError(
          domain: "BBMetalCameraErrorDomain", code: 0,
          userInfo: [NSLocalizedDescriptionKey: "Can not get Metal texture"])
        result = .failure(error)
      }
      let info = BBMetalFilterCompletionInfo(
        result: result,
        sampleTime: sampleTime,
        cameraPosition: cameraPosition,
        isCameraPhoto: isCameraPhoto)
      completion(info)
    }

    willTransmit?(texture.metalTexture, sampleTime)
    let output = BBMetalDefaultTexture(
      metalTexture: texture.metalTexture,
      sampleTime: sampleTime,
      cameraPosition: cameraPosition,
      isCameraPhoto: isCameraPhoto,
      cvMetalTexture: texture.cvMetalTexture)
    for consumer in consumers { consumer.newTextureAvailable(output, from: self) }

    // Benchmark
    if startTime != 0 {
      lock.wait()
      capturedFrameCount += 1
      if capturedFrameCount > ignoreInitialFrameCount {
        totalCaptureFrameTime += CACurrentMediaTime() - startTime
      }
      lock.signal()
    }
  }

  public func captureOutput(
    _ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    print("Camera drops \(output is AVCaptureAudioDataOutput ? "audio" : "video") sample buffer")
  }

  private func texture(with sampleBuffer: CMSampleBuffer) -> BBMetalVideoTextureItem? {
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
    return texture(with: imageBuffer)
  }

  private func texture(with imageBuffer: CVPixelBuffer) -> BBMetalVideoTextureItem? {
    let width = CVPixelBufferGetWidth(imageBuffer)
    let height = CVPixelBufferGetHeight(imageBuffer)

    #if !targetEnvironment(simulator)
      var cvMetalTextureOut: CVMetalTexture?
      let result = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        textureCache,
        imageBuffer,
        nil,
        .bgra8Unorm,  // camera ouput BGRA
        width,
        height,
        0,
        &cvMetalTextureOut)
      if result == kCVReturnSuccess,
        let cvMetalTexture = cvMetalTextureOut,
        let texture = CVMetalTextureGetTexture(cvMetalTexture)
      {
        return BBMetalVideoTextureItem(metalTexture: texture, cvMetalTexture: cvMetalTexture)
      }
    #endif
    return nil
  }
}

extension BBMetalCamera: AVCapturePhotoCaptureDelegate {
  public func photoOutput(
    _ output: AVCapturePhotoOutput,
    didFinishProcessingPhoto photo: AVCapturePhoto,
    error: Error?
  ) {
    guard let delegate = photoDelegate else { return }

    if let error = error {
      delegate.camera(self, didFail: error)
      return
    }

    guard let pixelBuffer = photo.pixelBuffer,
      let texture = texture(with: pixelBuffer),
      // Setting `videoOrientation` of `AVCaptureConnection` dose not work. So rotate texture here.
      let rotatedTexture = rotatedTexture(with: texture.metalTexture, angle: 90)
    else {
      delegate.camera(
        self,
        didFail: NSError(
          domain: "BBMetalCamera.Photo", code: 0,
          userInfo: [NSLocalizedDescriptionKey: "Can not get Metal texture"]))
      return
    }

    delegate.camera(self, didOutput: rotatedTexture)
  }

  private func rotatedTexture(with inTexture: MTLTexture, angle: Float) -> MTLTexture? {
    let source = BBMetalStaticImageSource(texture: inTexture)
    let filter = BBMetalRotateFilter(angle: angle, fitSize: true)
    source.add(consumer: filter).runSynchronously = true
    source.transmitTexture()
    return filter.outputTexture
  }
}

extension BBMetalCamera: AVCaptureMetadataOutputObjectsDelegate {
  public func metadataOutput(
    _ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject],
    from connection: AVCaptureConnection
  ) {
    metadataObjectDelegate?.camera(self, didOutput: metadataObjects)
  }
}

extension BBMetalCamera {
  class DeviceLookup {
    let session: AVCaptureDevice.DiscoverySession

    private let deviceTypes: [AVCaptureDevice.DeviceType]

    // TODO: Remove when device lookup issue fix is verified
    var internalErrorHandler: ((BBMetalCameraInternalError) -> Void)?

    init() {
      // Order matters: `device(for:)` picks the earliest match, preferring the richest device.
      let deviceTypes: [AVCaptureDevice.DeviceType] = [
        .builtInTripleCamera,
        .builtInDualWideCamera,
        .builtInDualCamera,
        .builtInWideAngleCamera
      ]

      self.deviceTypes = deviceTypes
      self.session = AVCaptureDevice.DiscoverySession(
        deviceTypes: deviceTypes,
        mediaType: .video,
        position: .unspecified
      )
    }

    func device(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
      let oldDeviceImpl = _oldImplDevice(for: position)
      let newDeviceImpl = _newImplDevice(for: position)

      if oldDeviceImpl == nil {
        internalErrorHandler?(.noVideoDeviceOldImpl)
      }

      if newDeviceImpl == nil {
        internalErrorHandler?(.noVideoDeviceNewImpl)
      }

      return oldDeviceImpl ?? newDeviceImpl
    }

    // TODO: Move code directly to `device(for:)` when device lookup issue fix is verified
    private func _newImplDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
      let candidates = session.devices
        .filter({ $0.position == position })
        .sorted(by: {
          let lhsIdx = deviceTypes.firstIndex(of: $0.deviceType) ?? 0
          let rhsIdx = deviceTypes.firstIndex(of: $1.deviceType) ?? 0
          return lhsIdx < rhsIdx
        })

      return candidates.first
    }

    // TODO: Remove when device lookup issue fix is verified
    private func _oldImplDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
      guard #available(iOS 18, *) else {
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
      }

      if position == .front {
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
      }

      if let device = AVCaptureDevice.default(
        .builtInTripleCamera,
        for: .video,
        position: position
      ) {
        return device
      } else if let device = AVCaptureDevice.default(
        .builtInDualWideCamera,
        for: .video,
        position: position
      ) {
        return device
      } else if let device = AVCaptureDevice.default(
        .builtInDualCamera,
        for: .video,
        position: position
      ) {
        return device
      } else {
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
      }
    }
  }
}
