//
//  BBMetalMultipleCamera.swift
//  BBMetalImage
//
//  Created by Kaibo Lu on 10/5/21.
//  Copyright © 2021 Kaibo Lu. All rights reserved.
//

import UIKit
import AVFoundation

@available(iOS 13.0, *)
public class BBMetalMultipleCamera {

    public static var isMultiCamSupported: Bool { AVCaptureMultiCamSession.isMultiCamSupported }
    
    private let session: AVCaptureMultiCamSession
    
    public let backCamera: BBMetalCamera
    public let frontCamera: BBMetalCamera
    
    public init() throws {
        session = .init()

        // `multitpleSessions: true` keeps the microphone in its own capture session. Attaching it to
        // the multi-cam session instead would reconfigure a live session at the moment recording
        // starts, and re-provisioning multi-cam hardware makes both previews re-run their exposure
        // ramp — visible as a freeze and a darkening pass over each camera in turn.
        self.backCamera = try BBMetalCamera(
            captureSession: session,
            position: .back,
            multitpleSessions: true
        )
        self.frontCamera = try BBMetalCamera(
            captureSession: session,
            position: .front,
            multitpleSessions: true
        )
    }

}
