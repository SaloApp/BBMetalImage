//
//  PiPFilter.swift
//  BBMetalImageDemo
//
//  Created by Kaibo Lu on 10/5/21.
//  Copyright © 2021 Kaibo Lu. All rights reserved.
//

import UIKit
import BBMetalImage

class PiPFilter: BBMetalBaseFilter {

    var pipFrame: SIMD4<Float>
    
    init(pipFrame: SIMD4<Float> = .zero) {
        self.pipFrame = pipFrame
        super.init(kernelFunctionName: "pipKernel", useMainBundleKernel: true)
    }
    
    override func updateParameters(forComputeCommandEncoder encoder: MTLComputeCommandEncoder) {
        encoder.setBytes(&pipFrame, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
    }

}
